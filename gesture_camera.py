"""Run live webcam gesture recognition with the bundled MediaPipe model.

The helpers in this module (camera setup, recognizer setup, frame loop and
hand drawing) are shared by the iterations in trash_talk.py and
music_control.py.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "my_custom_gesture"
    / "gesture_recognizer.task"
)
WINDOW_NAME = "Image"


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
