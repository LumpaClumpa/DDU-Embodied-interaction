import mediapipe as mp
import cv2
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640) #bredde
cap.set(4, 480) #højde

def main():
    with mp_hands.Hands(
        max_num_hands=2, #mængden af hænder der kan detekteres
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    ) as hands:
    

        if not cap.isOpened():
            print("Failed to open camera")
            return

        try:
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 480, 360)

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
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            img,
                            hand_landmarks, mp_hands.HAND_CONNECTIONS
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