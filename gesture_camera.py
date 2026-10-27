"""Run live webcam gesture recognition with the bundled MediaPipe model.

The helpers in this module (camera setup, recognizer setup, frame loop and
hand drawing) are shared by the iterations in trash_talk.py and
music_control.py.
"""

import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "my_custom_gesture"
    / "gesture_recognizer.task"
)
WINDOW_NAME = "Image"

# Clean sans-serif fonts for on-screen text, tried in order. OpenCV's own
# Hershey fonts look hand-drawn, so text is drawn with Pillow instead.
FONT_FILES = {
    False: ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
    True: ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
}
_font_cache = {}


def create_recognizer(model_path=MODEL_PATH, num_hands=2):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Gesture recognizer model not found: {model_path}")

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return vision.GestureRecognizer.create_from_options(options)


def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def camera_frames(cap, recognizer, window_size=(480, 360)):
    """Yield (mirrored BGR frame, recognizer results) until q is pressed.

    The caller draws on the frame; it is shown when the next frame is requested.
    """
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *window_size)
    last_timestamp_ms = -1

    while True:
        success, img = cap.read()
        attempt = 0
        while not success and attempt < 5:
            time.sleep(0.2)
            success, img = cap.read()
            attempt += 1

        if not success:
            print("Failed to read frame")
            return

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            last_timestamp_ms + 1,
        )
        last_timestamp_ms = timestamp_ms
        results = recognizer.recognize_for_video(mp_image, timestamp_ms)

        yield img, results

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def run_camera_app(
    handle_frame,
    model_path=MODEL_PATH,
    window_size=(480, 360),
    num_hands=2,
):
    """Open the camera and call handle_frame(img, results) for every frame."""
    cap = open_camera()
    try:
        if not cap.isOpened():
            print("Failed to open camera")
            return

        with create_recognizer(model_path, num_hands) as recognizer:
            for img, results in camera_frames(cap, recognizer, window_size):
                handle_frame(img, results)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def top_gesture(results, hand_index):
    """Return the best gesture for a hand, or None."""
    if hand_index < len(results.gestures) and results.gestures[hand_index]:
        return results.gestures[hand_index][0]
    return None


def hand_label(results, hand_index):
    """Return "Left" or "Right" as seen in the mirrored frame, or "Hand"."""
    if hand_index < len(results.handedness) and results.handedness[hand_index]:
        return results.handedness[hand_index][0].category_name
    return "Hand"


def _font(size, bold):
    key = (size, bold)
    if key not in _font_cache:
        fonts_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        font = None
        for name in FONT_FILES[bold]:
            for candidate in (fonts_dir / name, name):
                try:
                    font = ImageFont.truetype(str(candidate), size)
                    break
                except OSError:
                    continue
            if font is not None:
                break
        _font_cache[key] = font or ImageFont.load_default(size=size)
    return _font_cache[key]


def text_size(text, size=18, bold=False):
    """Return (width, height) of text drawn with draw_text."""
    left, top, right, bottom = _font(size, bold).getbbox(text)
    return right - left, bottom - top


def draw_text(img, text, org, size=18, color=(0, 0, 0), bold=False,
              background=None, padding=5):
    """Draw text with its top-left corner at org.

    Colors are BGR. With background, a filled box is drawn behind the text.
    Only the small area around the text is converted for Pillow, so this is
    cheap enough to call several times per frame.
    """
    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(text)
    x, y = org[0] - left, org[1] - top
    img_h, img_w = img.shape[:2]
    x0, y0 = max(org[0] - padding, 0), max(org[1] - padding, 0)
    x1 = min(org[0] + right - left + padding, img_w)
    y1 = min(org[1] + bottom - top + padding, img_h)
    if x0 >= x1 or y0 >= y1:
        return

    if background is not None:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), background, -1)
    patch = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]))
    ImageDraw.Draw(patch).text((x - x0, y - y0), text, font=font, fill=tuple(color[::-1]))
    img[y0:y1, x0:x1] = np.asarray(patch)[:, :, ::-1]


def draw_hand(img, hand_landmarks):
    vision.drawing_utils.draw_landmarks(
        img,
        hand_landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
    )


def draw_fingertips(img, hand_landmarks):
    h, w, _ = img.shape
    finger_tips = {
        "Thumb": hand_landmarks[4],
        "Index": hand_landmarks[8],
        "Middle": hand_landmarks[12],
        "Ring": hand_landmarks[16],
        "Pinky": hand_landmarks[20],
    }

    for name, landmark in finger_tips.items():
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.putText(
            img,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


def draw_gesture_label(img, hand_landmarks, gesture):
    h, w, _ = img.shape
    label = f"{gesture.category_name} ({gesture.score:.2f})"
    top_y = min(int(point.y * h) for point in hand_landmarks)
    left_x = min(int(point.x * w) for point in hand_landmarks)
    cv2.putText(
        img,
        label,
        (left_x, max(top_y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _draw_basic_frame(img, results):
    for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
        draw_hand(img, hand_landmarks)
        draw_fingertips(img, hand_landmarks)

        gesture = top_gesture(results, hand_index)
        if gesture is not None:
            draw_gesture_label(img, hand_landmarks, gesture)


def run_gesture_camera():
    run_camera_app(_draw_basic_frame)
