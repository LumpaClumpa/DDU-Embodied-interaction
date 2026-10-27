"""Iteration 1: friendly trash talk triggered by recognized gestures.

Hold a gesture for a moment and the program "says" a message for it: the
message is printed to the console and shown in a speech banner on screen.
Players are named after the side of the screen their hand is on, so two
people can trash talk each other in front of one camera.

Run:
    python trash_talk.py
    python trash_talk.py --model models\\test_model\\gesture_recognizer.task
"""

import argparse
import random
import textwrap
import time
from pathlib import Path

import cv2

from gesture_camera import (
    MODEL_PATH,
    draw_hand,
    run_camera_app,
    top_gesture,
)


# Keys are lowercase gesture labels. The first block matches the custom model;
# the second block matches the gestures in MediaPipe's stock model, so other
# models also work.
MESSAGES = {
    "nice!": [
        "{player}: Nice! ...said nobody about your last move.",
        "{player}: Nice! Now do that again, but good.",
        "{player}: Nice! I'm being sarcastic, by the way.",
        "{player}: Nice! You almost looked like you knew what you were doing.",
    ],
    "flipoff": [
        "{player}: This one's for you, buddy. With love.",
        "{player}: Consider yourself officially roasted.",
        "{player}: Sending you my warmest regards.",
        "{player}: That's my review of your skills.",
    ],
    "thumb_up": [
        "{player}: Great job! For a beginner.",
        "{player}: Thumbs up for showing up. That's about it.",
    ],
    "thumb_down": [
        "{player}: Rated: needs work. Lots of work.",
        "{player}: Boo! Do better!",
    ],
    "victory": [
        "{player}: Victory is mine. Again.",
        "{player}: Peace out, loser!",
    ],
    "pointing_up": [
        "{player}: Number one. And it's not you.",
        "{player}: Look up there, that's where my score is.",
    ],
    "open_palm": [
        "{player}: Talk to the hand.",
        "{player}: High five? Nah, just waving goodbye to your chances.",
    ],
    "closed_fist": [
        "{player}: Fist bump? Earn it first.",
        "{player}: Rock. Because you play like one.",
    ],
    "iloveyou": [
        "{player}: Love you, but you're still losing.",
        "{player}: Hugs and kisses... and an L.",
    ],
}
FALLBACK_MESSAGE = "{player} throws a '{gesture}'. Bold move."
IGNORED_GESTURES = {"none", ""}

MIN_SCORE = 0.6        # Ignore uncertain predictions.
HOLD_SECONDS = 0.6     # How long a gesture must be held before it triggers.
COOLDOWN_SECONDS = 2.0 # Minimum time between two messages.
BANNER_SECONDS = 4.0   # How long a message stays on screen.


class TrashTalker:
    def __init__(self):
        # Per player: the gesture being held, when it started and whether it
        # already triggered a message (a gesture triggers only once per hold).
        self.holds = {}
        self.last_message_time = 0.0
        self.last_messages = {}
        self.banner_text = ""
        self.banner_time = 0.0

    def pick_message(self, gesture_name, player):
        key = gesture_name.lower()
        options = MESSAGES.get(key)
        if not options:
            return FALLBACK_MESSAGE.format(player=player, gesture=gesture_name)

        # Avoid repeating the same line twice in a row.
        choices = [m for m in options if m != self.last_messages.get(key)]
        message = random.choice(choices or options)
        self.last_messages[key] = message
        return message.format(player=player)

    def say(self, message, now):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.banner_text = message
        self.banner_time = now
        self.last_message_time = now

    def handle_frame(self, img, results):
        now = time.monotonic()
        h, w, _ = img.shape
        seen_players = set()

        for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
            draw_hand(img, hand_landmarks)

            wrist = hand_landmarks[0]
            player = "Left player" if wrist.x < 0.5 else "Right player"
            seen_players.add(player)

            gesture = top_gesture(results, hand_index)
            name = gesture.category_name if gesture else ""
            if gesture is None or gesture.score < MIN_SCORE:
                name = ""

            hold = self.holds.get(player)
            if hold is None or hold["gesture"] != name:
                hold = {"gesture": name, "since": now, "fired": False}
                self.holds[player] = hold

            if name.lower() in IGNORED_GESTURES:
                continue

            progress = min((now - hold["since"]) / HOLD_SECONDS, 1.0)
            center = (int(wrist.x * w), int(wrist.y * h))
            self.draw_hold_ring(img, center, progress, name, hold["fired"])

            cooled_down = now - self.last_message_time >= COOLDOWN_SECONDS
            if progress >= 1.0 and not hold["fired"] and cooled_down:
                hold["fired"] = True
                self.say(self.pick_message(name, player), now)

        # A player whose hand left the frame starts a fresh hold next time.
        for player in list(self.holds):
            if player not in seen_players:
                del self.holds[player]

        self.draw_banner(img, now)

    @staticmethod
    def draw_hold_ring(img, center, progress, name, fired):
        color = (0, 200, 255) if not fired else (0, 255, 0)
        cv2.circle(img, center, 28, (60, 60, 60), 3)
        cv2.ellipse(img, center, (28, 28), -90, 0, int(360 * progress), color, 4)
        cv2.putText(
            img,
            name,
            (center[0] - 30, center[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def draw_banner(self, img, now):
        age = now - self.banner_time
        if not self.banner_text or age > BANNER_SECONDS:
            return

        h, w, _ = img.shape
        lines = textwrap.wrap(self.banner_text, width=40)
        line_height = 28
        box_h = line_height * len(lines) + 20
        top = h - box_h - 10

        # Fade the banner out during its last second.
        alpha = min(1.0, BANNER_SECONDS - age) * 0.75
        overlay = img.copy()
        cv2.rectangle(overlay, (10, top), (w - 10, h - 10), (40, 20, 60), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (10, top), (w - 10, h - 10), (200, 120, 255), 2)

        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line,
                (24, top + 30 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )


def main():
    parser = argparse.ArgumentParser(description="Gesture-triggered trash talk.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to a gesture_recognizer.task file.",
    )
    args = parser.parse_args()

    print("Hold a gesture to trash talk. Press q in the camera window to quit.")
    talker = TrashTalker()
    run_camera_app(talker.handle_frame, model_path=args.model, window_size=(640, 480))


if __name__ == "__main__":
    main()
