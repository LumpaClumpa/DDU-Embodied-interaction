"""Train a custom MediaPipe gesture recognizer from a labeled image folder.

Dataset layout:
    data/gesture_dataset/
        none/
            image_001.jpg
        thumbs_up/
            image_001.jpg
        peace/
            image_001.jpg

The ``none`` class is required and should contain hands making gestures outside
the custom classes. Each image should clearly show a hand.
"""

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import tensorflow as tf # pyright: ignore[reportMissingModuleSource]
from mediapipe_model_maker import gesture_recognizer # pyright: ignore[reportMissingImports]
from mediapipe_model_maker.python.core.data.classification_dataset import (
    ClassificationDataset,
)


SAMPLE_DATA_URL = (
    "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/"
    "rps_data_sample.zip"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR / "data" / "gesture_dataset",
        help="Folder containing one image subfolder per gesture label.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Directory in which to save gesture_recognizer.task. By default, "
            "a directory named after the dataset is created under models."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (default: 2).",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Download and train on the guide's Rock-Paper-Scissors sample data.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def download_sample_dataset() -> Path:
    sample_zip = PROJECT_DIR / "data" / "rps_data_sample.zip"
    sample_dir = PROJECT_DIR / "data" / "rps_data_sample"
    sample_zip.parent.mkdir(parents=True, exist_ok=True)
    if not sample_dir.is_dir():
        print(f"Downloading sample dataset from {SAMPLE_DATA_URL}")
        urllib.request.urlretrieve(SAMPLE_DATA_URL, sample_zip)
        with zipfile.ZipFile(sample_zip) as archive:
            data_dir = sample_zip.parent.resolve()
            for member in archive.infolist():
                extracted_path = (data_dir / member.filename).resolve()
                if (
                    extracted_path != data_dir
                    and data_dir not in extracted_path.parents
                ):
                    raise ValueError(
                        f"Unsafe path in sample dataset archive: {member.filename}"
                    )
            archive.extractall(sample_zip.parent)
    if not sample_dir.is_dir():
        raise FileNotFoundError(
            f"Sample archive did not contain the expected folder: {sample_dir}"
        )
    return sample_dir


def validate_dataset(dataset_dir: Path) -> dict[str, int]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create a folder per gesture label, or run with --sample-data."
        )

    label_counts = {
        folder.name: sum(
            1
            for image in folder.iterdir()
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
        )
        for folder in dataset_dir.iterdir()
        if folder.is_dir()
    }
    if not label_counts:
        raise ValueError(f"No gesture label folders found in {dataset_dir}.")
    if not any(label.lower() == "none" for label in label_counts):
        raise ValueError(
            "The dataset must include a 'none' folder for gestures outside "
            "your custom classes."
        )
    if len(label_counts) < 2:
        raise ValueError("Add at least one gesture folder alongside 'none'.")

    empty_labels = [label for label, count in label_counts.items() if count == 0]
    if empty_labels:
        raise ValueError(
            "These label folders contain no supported images "
            f"({', '.join(sorted(empty_labels))})."
        )
    return label_counts


def has_added_classes(export_dir: Path, label_names: list[str]) -> bool:
    labels_path = export_dir / "training_labels.json"
    checkpoint_dir = export_dir / "epoch_models"
    has_checkpoints = checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())
    if not labels_path.is_file():
        if has_checkpoints:
            raise FileExistsError(
                f"Training checkpoints already exist in {checkpoint_dir}, "
                "but there is no label manifest to verify them. Choose a new "
                "directory with --export-dir to start a clean training run."
            )
        return False

    previous_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(previous_labels, list) or not all(
        isinstance(label, str) for label in previous_labels
    ):
        raise ValueError(f"Invalid label manifest: {labels_path}")
    if previous_labels == label_names:
        return False

    added_labels = [label for label in label_names if label not in previous_labels]
    removed_labels = [label for label in previous_labels if label not in label_names]
    if not added_labels or removed_labels:
        raise ValueError(
            "The existing checkpoint has different labels "
            f"({previous_labels}); current dataset labels are {label_names}. "
            "Automatic retraining is only available when adding classes. "
            "Choose a new --export-dir for a clean training run."
        )

    print(f"New gesture classes detected: {', '.join(added_labels)}")
    print("Training a fresh classifier on all current classes.")
    return True


def replace_export_dir(staged_dir: Path, export_dir: Path) -> None:
    backup_parent = Path(
        tempfile.mkdtemp(prefix=f".{export_dir.name}-backup-", dir=export_dir.parent)
    )
    backup_dir = backup_parent / export_dir.name
    try:
        os.replace(export_dir, backup_dir)
        try:
            os.replace(staged_dir, export_dir)
        except OSError as install_error:
            try:
                os.replace(backup_dir, export_dir)
            except OSError as restore_error:
                raise RuntimeError(
                    f"Could not install the newly trained model or restore the "
                    f"previous one. The previous model is preserved at "
                    f"{backup_dir}."
                ) from restore_error
            raise install_error
        shutil.rmtree(backup_parent)
    except OSError:
        if backup_parent.exists() and not backup_dir.exists():
            shutil.rmtree(backup_parent)
        raise


def train_and_export(
    export_dir: Path,
    train_data: ClassificationDataset,
    validation_data: ClassificationDataset,
    test_data: ClassificationDataset,
    label_names: list[str],
    epochs: int,
    batch_size: int,
) -> Path:
    hparams = gesture_recognizer.HParams(
        export_dir=str(export_dir),
        epochs=epochs,
        batch_size=batch_size,
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    loss, accuracy = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss:.4f}; test accuracy: {accuracy:.4f}")

    from mediapipe.tasks.python.metadata.metadata_writers import (
        model_asset_bundle_utils,
        writer_utils,
    )

    if not hasattr(writer_utils, "create_model_asset_bundle"):
        # Model Maker 0.1.x uses the helper name from before MediaPipe 0.10.5.
        setattr(
            writer_utils,
            "create_model_asset_bundle",
            model_asset_bundle_utils.create_model_asset_bundle,
        )
    model.export_model()

    labels_path = export_dir / "training_labels.json"
    labels_path.write_text(
        json.dumps(label_names, indent=2) + "\n",
        encoding="utf-8",
    )
    return export_dir / "gesture_recognizer.task"


def train(dataset_dir: Path, export_dir: Path, epochs: int, batch_size: int) -> Path:
    label_counts = validate_dataset(dataset_dir)
    print("Images per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    data = gesture_recognizer.Dataset.from_folder(
        dirname=str(dataset_dir),
        hparams=gesture_recognizer.HandDataPreprocessingParams(
            shuffle=True,
            min_detection_confidence=0.7,
        ),
    )
    if data.size < 10:
        raise ValueError(
            f"Only {data.size} images with detected hands remain. "
            "Add more clear hand images before splitting the dataset."
        )

    train_data, remainder = data.split(0.8)
    validation_data, test_data = remainder.split(0.5)
    if not train_data.size or not validation_data.size or not test_data.size:
        raise ValueError(
            "The dataset is too small to create non-empty train, validation, "
            "and test splits."
        )

    replace_existing = has_added_classes(export_dir, data.label_names)
    if replace_existing:
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{export_dir.name}-training-",
            dir=export_dir.parent,
        ) as temporary_dir:
            staged_dir = Path(temporary_dir) / "model"
            train_and_export(
                staged_dir,
                train_data,
                validation_data,
                test_data,
                data.label_names,
                epochs,
                batch_size,
            )
            replace_export_dir(staged_dir, export_dir)
        print("Replaced the previous model with the updated all-class model.")
    else:
        export_dir.mkdir(parents=True, exist_ok=True)
        train_and_export(
            export_dir,
            train_data,
            validation_data,
            test_data,
            data.label_names,
            epochs,
            batch_size,
        )

    model_path = export_dir / "gesture_recognizer.task"
    print(f"Exported model: {model_path}")
    return model_path


def main() -> None:
    if not tf.__version__.startswith("2"):
        raise RuntimeError(f"TensorFlow 2 is required; found {tf.__version__}.")

    args = parse_args()
    dataset_dir = (
        download_sample_dataset() if args.sample_data else args.dataset.resolve()
    )
    export_dir = (
        args.export_dir.resolve()
        if args.export_dir
        else PROJECT_DIR / "models" / f"{dataset_dir.name}_gesture_recognizer"
    )
    train(
        dataset_dir=dataset_dir,
        export_dir=export_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
