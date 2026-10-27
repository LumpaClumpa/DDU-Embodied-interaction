"""Tutorial for music_control.py: learn the gestures level by level.

Each level teaches one mechanic and is complete when you have done it a few
times. A wrong move shows what the program saw instead, with a tip. The last
level is a timed challenge with random moves that gives one to three stars.

By default the tutorial does not send media keys, so practising does not
change your music. Add --live to control the music player while you play.

Keys in the camera window:
    n  skip to the next level      b  back to the previous level
    r  restart the current level   q  quit

Run:
    python music_tutorial.py
    python music_tutorial.py --live
    python music_tutorial.py --level 6   (start at level 6)
"""

import argparse
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from gesture_camera import MODEL_PATH, draw_hand, draw_text, run_camera_app, text_size
from media_sources import AudioSpectrum, NowPlaying
from music_control import (
    PANEL_COLOR,
    TEXT_COLOR,
    VISUALIZER_BANDS,
    MediaKeys,
    MusicController,
)


PRESENCE_SECONDS = 1.5      # How long a hand must be visible in level 1.
LEVEL_DONE_SECONDS = 1.5    # How long "Level complete" shows.
FEEDBACK_SECONDS = 2.5      # How long a tip after a wrong move shows.
CHALLENGE_MOVES = 6
CHALLENGE_GRACE = 0.8       # Ignore extra events right after a correct move,
                            # e.g. a second volume step from the same pinch.
CHALLENGE_POOL = ["PLAY/PAUSE", "NEXT", "PREVIOUS", "VOLUME UP", "VOLUME DOWN"]
# (max seconds, max mistakes) for three and two stars.
STAR_LIMITS = ((25, 0), (40, 2))

SUCCESS_COLOR = (60, 170, 60)
WRONG_COLOR = (40, 40, 200)

# Short instruction for each move, used by the challenge and wrong-move tips.
HOW_TO = {
    "PLAY/PAUSE": "make a fist",
    "NEXT": "swipe your open hand right",
    "PREVIOUS": "swipe your open hand left",
    "VOLUME UP": "pinch and move up",
    "VOLUME DOWN": "pinch and move down",
    "VISUALIZER PLACED": "frame with both hands, drag apart, relax",
    "VISUALIZER CLOSED": "make a small frame, then relax",
}

LEVELS = [
    {
        "kind": "presence",
        "title": "Say hi",
        "text": "Hold one hand up so the camera can see it.",
        "tip": "Keep your whole hand inside the picture.",
    },
    {
        "kind": "action", "goal": "PLAY/PAUSE", "count": 2,
        "title": "Play / pause",
        "text": "Close your hand into a fist to play or pause.",
        "tip": "Curl all four fingers and tuck your thumb in.",
    },
    {
        "kind": "action", "goal": "NEXT", "count": 2,
        "title": "Next track",
        "text": "Open your hand, wait a moment, then swipe fast to the right.",
        "tip": "Watch the bar at the top: green means the swipe counts.",
    },
    {
        "kind": "action", "goal": "PREVIOUS", "count": 2,
        "title": "Previous track",
        "text": "Open your hand, wait a moment, then swipe fast to the left.",
        "tip": "Bringing your hand back after a swipe does not count. Pause first.",
    },
    {
        "kind": "action", "goal": "VOLUME UP", "count": 3,
        "title": "Volume up",
        "text": "Pinch thumb and index finger together, then move your hand up.",
        "tip": "Keep pinching while you move. Each small step is one volume step.",
    },
    {
        "kind": "action", "goal": "VOLUME DOWN", "count": 3,
        "title": "Volume down",
        "text": "Pinch thumb and index finger together, then move your hand down.",
        "tip": "Start the pinch high, then lower your hand slowly.",
    },
    {
        "kind": "action", "goal": "VISUALIZER PLACED", "count": 1,
        "title": "Open the visualizer",
        "text": "Make an L with thumb and index finger on both hands, drag them "
                "apart, then relax your hands.",
        "tip": "Hold both L shapes still for a moment before you drag.",
    },
    {
        "kind": "action", "goal": "VISUALIZER CLOSED", "count": 1,
        "title": "Close the visualizer",
        "text": "Make the frame again with your hands close together, then relax.",
        "tip": "The corners turn red when the frame is small enough to close it.",
    },
    {
        "kind": "challenge",
        "title": "Final challenge",
        "text": f"Do the {CHALLENGE_MOVES} moves shown as fast as you can.",
        "tip": "Mistakes cost stars. Take a breath between moves.",
    },
    {
        "kind": "done",
        "title": "Tutorial complete",
        "text": "You know every move. Run music_control.py to use them for real.",
        "tip": "",
    },
]


def normalize(action):
    return "PLAY/PAUSE" if action in ("PLAY", "PAUSE") else action


class TutorialController(MusicController):
    """MusicController that also records every action it triggers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def trigger(self, action, vk_code, now, presses=1):
        self.events.append(normalize(action))
        super().trigger(action, vk_code, now, presses)


class Tutorial:
    def __init__(self, controller, start_level=0):
        self.controller = controller
        self.issue_seen = 0.0  # swipe_issue_time of the last explained swipe.
        self.start_level(min(max(start_level, 0), len(LEVELS) - 1))

    # ---- Level state -----------------------------------------------------

    def start_level(self, index):
        self.level = index
        self.progress = 0
        self.done_time = None
        self.feedback = ""
        self.feedback_time = 0.0
        self.feedback_good = False
        self.hand_since = None
        # Challenge state.
        self.prompts = self.make_prompts()
        self.prompt_index = 0
        self.mistakes = 0
        self.challenge_start = None
        self.challenge_time = None
        self.grace_until = 0.0
        self.stars = 0

    @staticmethod
    def make_prompts():
        prompts = []
        while len(prompts) < CHALLENGE_MOVES:
            move = random.choice(CHALLENGE_POOL)
            if not prompts or move != prompts[-1]:
                prompts.append(move)
        return prompts

    def on_key(self, key):
        if key == "n":
            self.start_level(min(self.level + 1, len(LEVELS) - 1))
        elif key == "b":
            self.start_level(max(self.level - 1, 0))
        elif key == "r":
            self.start_level(self.level)

    def say(self, text, now, good=False):
        self.feedback, self.feedback_time, self.feedback_good = text, now, good

    # ---- Frame handling --------------------------------------------------

    def handle_frame(self, img, results):
        now = time.monotonic()
        for hand in results.hand_landmarks:
            draw_hand(img, hand)

        self.controller.events.clear()
        self.controller.handle_frame(img, results)
        events = list(self.controller.events)

        level = LEVELS[self.level]
        if self.done_time is not None:
            if now - self.done_time >= LEVEL_DONE_SECONDS:
                self.start_level(self.level + 1)
        elif level["kind"] == "presence":
            self.update_presence(results, now)
        elif level["kind"] == "action":
            self.update_action(level, events, now)
        elif level["kind"] == "challenge":
            self.update_challenge(events, now)
        self.explain_swipe(level, events, now)

        self.draw(img, now)

    def explain_swipe(self, level, events, now):
        """When a swipe is the move asked for and an attempt did not count,
        say why (too slow, diagonal, ...)."""
        issue_time = self.controller.swipe_issue_time
        if issue_time <= self.issue_seen:
            return
        self.issue_seen = issue_time
        if self.done_time is not None or events:
            return
        if level["kind"] == "action":
            wanted = level["goal"]
        elif level["kind"] == "challenge" and self.challenge_time is None:
            wanted = self.prompts[self.prompt_index]
        else:
            return
        if wanted in ("NEXT", "PREVIOUS"):
            self.say(f"Almost! {self.controller.swipe_issue}", now)

    def complete(self, now):
        self.done_time = now

    def update_presence(self, results, now):
        if len(results.hand_landmarks) == 1:
            if self.hand_since is None:
                self.hand_since = now
            if now - self.hand_since >= PRESENCE_SECONDS:
                self.complete(now)
        else:
            self.hand_since = None

    def update_action(self, level, events, now):
        for event in events:
            if event == level["goal"]:
                self.progress += 1
                self.say("Nice!", now, good=True)
                if self.progress >= level["count"]:
                    self.complete(now)
                    return
            else:
                self.say(f"That was {event}. For {level['title']}: "
                         f"{HOW_TO[level['goal']]}.", now)

    def update_challenge(self, events, now):
        if self.challenge_time is not None:
            return
        if self.challenge_start is None:
            self.challenge_start = now
        for event in events:
            if now < self.grace_until:
                continue
            if event == self.prompts[self.prompt_index]:
                self.prompt_index += 1
                self.grace_until = now + CHALLENGE_GRACE
                self.say("Correct!", now, good=True)
                if self.prompt_index >= len(self.prompts):
                    self.challenge_time = now - self.challenge_start
                    self.stars = self.rate(self.challenge_time, self.mistakes)
                    self.complete(now + 1.5)  # Show the result a bit longer.
                    return
            else:
                self.mistakes += 1
                self.grace_until = now + CHALLENGE_GRACE
                self.say(f"That was {event}. Wanted: "
                         f"{HOW_TO[self.prompts[self.prompt_index]]}.", now)

    @staticmethod
    def rate(seconds, mistakes):
        stars = 1
        for limit_seconds, limit_mistakes in reversed(STAR_LIMITS):
            if seconds <= limit_seconds and mistakes <= limit_mistakes:
                stars += 1
        return stars

    # ---- Drawing ---------------------------------------------------------

    def draw(self, img, now):
        h, w, _ = img.shape
        level = LEVELS[self.level]

        # Bottom panel: level number, title, instruction, progress, keys.
        panel_top = h - 118
        overlay = img.copy()
        cv2.rectangle(overlay, (0, panel_top), (w, h), PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

        draw_text(img, f"LEVEL {self.level + 1} / {len(LEVELS)}", (16, panel_top + 10), 12,
                  TEXT_COLOR, bold=True)
        draw_text(img, level["title"], (16, panel_top + 26), 22, TEXT_COLOR, bold=True)
        for i, line in enumerate(wrap(level["text"], 13, w - 32)[:2]):
            draw_text(img, line, (16, panel_top + 58 + i * 18), 13, TEXT_COLOR)
        keys = "n skip  ·  b back  ·  r restart  ·  q quit"
        keys_w, _ = text_size(keys, 11)
        draw_text(img, keys, (w - keys_w - 16, panel_top + 10), 11, (90, 90, 90))
        self.draw_progress(img, level, (w - 16, panel_top + 40), now)

        if level["kind"] == "challenge":
            self.draw_challenge(img, now)
        if level["kind"] == "done":
            self.draw_center(img, "Well done!", "You finished the tutorial.")

        if self.feedback and now - self.feedback_time < FEEDBACK_SECONDS:
            color = SUCCESS_COLOR if self.feedback_good else WRONG_COLOR
            lines = wrap(self.feedback, 15, w - 64)[:2]
            for i, line in enumerate(lines):
                line_w, _ = text_size(line, 15, bold=True)
                draw_text(img, line, ((w - line_w) // 2, panel_top - 34 - (len(lines) - 1 - i) * 26),
                          15, color, bold=True, background=PANEL_COLOR, padding=6)

        if self.done_time is not None and level["kind"] != "done":
            if level["kind"] == "challenge":
                self.draw_center(img, "Challenge complete!", f"{self.challenge_time:.1f} s  ·  "
                                 f"{self.mistakes} mistake{'s' * (self.mistakes != 1)}")
                draw_stars(img, self.stars, (w // 2, h // 2 - 120))
            else:
                self.draw_center(img, "Level complete!", "")

    def draw_progress(self, img, level, top_right, now):
        """Dots for counted moves, a bar for the presence level."""
        x1, y = top_right
        if level["kind"] == "action":
            done = level["count"] if self.done_time is not None else self.progress
            for i in range(level["count"]):
                cx = x1 - 12 - (level["count"] - 1 - i) * 30
                cv2.circle(img, (cx, y), 10, TEXT_COLOR, 2)
                if i < done:
                    cv2.circle(img, (cx, y), 7, SUCCESS_COLOR, -1)
        elif level["kind"] == "presence":
            fill = 1.0 if self.done_time is not None else (
                0.0 if self.hand_since is None
                else min((now - self.hand_since) / PRESENCE_SECONDS, 1.0))
            cv2.rectangle(img, (x1 - 120, y - 6), (x1, y + 6), TEXT_COLOR, 2)
            cv2.rectangle(img, (x1 - 120, y - 6), (x1 - 120 + int(120 * fill), y + 6),
                          SUCCESS_COLOR, -1)

    def draw_challenge(self, img, now):
        if self.challenge_time is not None:
            return
        h, w, _ = img.shape
        target = self.prompts[self.prompt_index]
        elapsed = 0.0 if self.challenge_start is None else now - self.challenge_start
        info = (f"Move {self.prompt_index + 1} / {len(self.prompts)}   "
                f"{elapsed:.1f} s   {self.mistakes} mistakes")
        info_w, _ = text_size(info, 13)
        draw_text(img, info, ((w - info_w) // 2, 52), 13, TEXT_COLOR, background=PANEL_COLOR)
        self.draw_center(img, target, HOW_TO[target])

    @staticmethod
    def draw_center(img, big, small):
        h, w, _ = img.shape
        big_w, big_h = text_size(big, 34, bold=True)
        y = h // 2 - 70
        draw_text(img, big, ((w - big_w) // 2, y), 34, TEXT_COLOR, bold=True,
                  background=PANEL_COLOR, padding=12)
        if small:
            small_w, _ = text_size(small, 15)
            draw_text(img, small, ((w - small_w) // 2, y + big_h + 22), 15, TEXT_COLOR,
                      background=PANEL_COLOR, padding=6)


def draw_stars(img, earned, center, radius=26):
    """Three stars in a row: filled gold for earned ones, outlined otherwise.
    Drawn as shapes because the text font has no star character."""
    cx, cy = center
    for i in range(3):
        x = cx + (i - 1) * (radius * 2 + 14)
        points = np.array([
            (x + (radius if k % 2 == 0 else radius * 0.45) * math.sin(math.pi * k / 5),
             cy - (radius if k % 2 == 0 else radius * 0.45) * math.cos(math.pi * k / 5))
            for k in range(10)
        ], dtype=np.int32)
        cv2.circle(img, (x, cy), radius + 8, PANEL_COLOR, -1)
        if i < earned:
            cv2.fillPoly(img, [points], (0, 190, 255))
        cv2.polylines(img, [points], True, TEXT_COLOR, 2)


def wrap(text, size, max_width):
    """Split text into lines that fit max_width pixels."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and text_size(candidate, size)[0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Learn the music_control gestures.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="Path to a gesture_recognizer.task file.")
    parser.add_argument("--live", action="store_true",
                        help="Send media keys to the music player while practising.")
    parser.add_argument("--level", type=int, default=1,
                        help=f"Level to start at (1-{len(LEVELS)}).")
    args = parser.parse_args()

    print("Music control tutorial. Keys: n next, b back, r restart, q quit.")
    audio, now_playing = AudioSpectrum(VISUALIZER_BANDS), NowPlaying()
    controller = TutorialController(MediaKeys(dry_run=not args.live), audio, now_playing)
    tutorial = Tutorial(controller, start_level=args.level - 1)
    try:
        run_camera_app(
            tutorial.handle_frame,
            model_path=args.model,
            window_size=(800, 600),
            num_hands=2,
            on_key=tutorial.on_key,
        )
    finally:
        audio.stop()
        now_playing.stop()


if __name__ == "__main__":
    main()
