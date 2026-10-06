import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640) #bredde
cap.set(4, 480) #højde

def main():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        if not cap.isOpened():
            print("Failed to open camera")
            return

        try:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 480, 360)
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
                    break

                img = cv2.flip(img, 1) #flipper billedet så det er ligesom et spejl
                h, w, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        vision.drawing_utils.draw_landmarks(
                            img,
                            hand_landmarks,
                            vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        )

                        finger_tips = {
                            "Thumb": hand_landmarks[4],
                            "Index": hand_landmarks[8],
                            "Middle": hand_landmarks[12],
                            "Ring": hand_landmarks[16],
                            "Pinky": hand_landmarks[20]
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
                                1
                            )

                            cv2.circle(
                                img,
                                (x, y),
                                5,
                                (0, 255, 0),
                                -1
                            )
                            
                cv2.imshow("Image", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

 
if __name__ == "__main__":
    main()