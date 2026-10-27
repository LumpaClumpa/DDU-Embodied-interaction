"""Iteration 2: control music playback with your hand and see it on screen.

Iteration 1 reacts to the gesture class from the model. This iteration reads
the hand landmarks (finger positions and hand movement) directly instead:

    Hold a closed fist             -> play / pause
    Swipe hand right / left        -> next / previous track
    Pinch thumb + index, move up   -> volume up
    Pinch thumb + index, move down -> volume down

Commands are sent as Windows media keys, so they control whatever player is
active (Spotify, YouTube in a browser, Windows Media Player, ...).

Run:
    python music_control.py
    python music_control.py --dry-run   (show actions without sending keys)
"""

import argparse
import ctypes
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from gesture_camera import MODEL_PATH, draw_hand, run_camera_app


HOLD_SECONDS = 0.6      # How long a fist must be held for play/pause.

SWIPE_DISTANCE = 0.30   # Fraction of frame width the wrist must travel...
SWIPE_WINDOW = 0.40     # ...within this many seconds.
SWIPE_COOLDOWN = 1.0

PINCH_RATIO = 0.35      # Pinch when tip distance < this * hand size.
VOLUME_STEP = 0.04      # Fraction of frame height per volume key press.

ACTION_FLASH_SECONDS = 0.8
TRAIL_LENGTH = 25

VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


class MediaKeys:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run or sys.platform != "win32"
        if self.dry_run and not dry_run:
            print("Media keys need Windows; running in dry-run mode.")

    def press(self, vk_code, name):
        print(f"[{time.strftime('%H:%M:%S')}] {name}")
        if self.dry_run:
            return
        user32 = ctypes.windll.user32
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_fist(hand):
    """True when all four fingertips are curled closer to the wrist than
    their middle joints. Landmark ids: tips 8/12/16/20, joints 6/10/14/18."""
    wrist = hand[0]
    return all(
        distance(wrist, hand[tip]) < distance(wrist, hand[joint])
        for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18))
    )


class MusicController:
    def __init__(self, keys):
        self.keys = keys
        self.playing = False  # Assumed state: we cannot read the real player.
        self.volume_steps = 0  # Net volume steps sent since start.

        self.hold_gesture = ""
        self.hold_since = 0.0
        self.hold_fired = False

        self.wrist_history = deque()
        self.last_swipe = 0.0

        self.pinch_anchor_y = None

        self.trail = deque(maxlen=TRAIL_LENGTH)
        self.action = ""
        self.action_time = 0.0
        self.motion_energy = 0.0
        self.start_time = time.monotonic()

    def trigger(self, action, vk_code, now):
        self.keys.press(vk_code, action)
        self.action = action
        self.action_time = now

    def handle_frame(self, img, results):
        now = time.monotonic()
        h, w, _ = img.shape

        if not results.hand_landmarks:
            self.wrist_history.clear()
            self.pinch_anchor_y = None
            self.hold_gesture = ""
            self.trail.clear()
            self.motion_energy *= 0.9
            self.draw_hud(img, now, None)
            return

        hand = results.hand_landmarks[0]
        draw_hand(img, hand)
        wrist, thumb_tip, index_tip = hand[0], hand[4], hand[8]
        hand_size = max(distance(wrist, hand[9]), 1e-6)
        fist = is_fist(hand)
        # A curled index finger resting on the thumb (a fist, or a gesture
        # like flipOff) is not a pinch: the index tip must be past its middle
        # joint (landmark 6).
        index_out = distance(wrist, index_tip) > distance(wrist, hand[6])
        pinching = (
            not fist
            and index_out
            and distance(thumb_tip, index_tip) < PINCH_RATIO * hand_size
        )

        self.update_motion(index_tip, w, h)
        self.update_play_pause(fist, now)
        if pinching:
            self.update_volume(thumb_tip, index_tip, now)
            self.wrist_history.clear()
        else:
            self.pinch_anchor_y = None
            self.update_swipe(wrist, now)

        self.draw_hud(img, now, (hand, pinching))

    def update_motion(self, index_tip, w, h):
        point = (int(index_tip.x * w), int(index_tip.y * h))
        if self.trail:
            last = self.trail[-1]
            speed = math.hypot(point[0] - last[0], point[1] - last[1])
            self.motion_energy = 0.8 * self.motion_energy + 0.2 * min(speed / 30, 1.0)
        self.trail.append(point)

    def update_play_pause(self, fist, now):
        name = "fist" if fist else ""
        if name != self.hold_gesture:
            self.hold_gesture = name
            self.hold_since = now
            self.hold_fired = False
        elif name and not self.hold_fired and now - self.hold_since >= HOLD_SECONDS:
            self.hold_fired = True
            self.playing = not self.playing
            self.trigger("PLAY" if self.playing else "PAUSE", VK_MEDIA_PLAY_PAUSE, now)

    def update_swipe(self, wrist, now):
        self.wrist_history.append((now, wrist.x))
        while self.wrist_history and now - self.wrist_history[0][0] > SWIPE_WINDOW:
            self.wrist_history.popleft()

        if now - self.last_swipe < SWIPE_COOLDOWN or len(self.wrist_history) < 3:
            return
        travel = wrist.x - self.wrist_history[0][1]
        if abs(travel) >= SWIPE_DISTANCE:
            self.last_swipe = now
            self.wrist_history.clear()
            if travel > 0:
                self.trigger("NEXT", VK_MEDIA_NEXT_TRACK, now)
            else:
                self.trigger("PREVIOUS", VK_MEDIA_PREV_TRACK, now)

    def update_volume(self, thumb_tip, index_tip, now):
        pinch_y = (thumb_tip.y + index_tip.y) / 2
        if self.pinch_anchor_y is None:
            self.pinch_anchor_y = pinch_y
            return

        # Moving up makes y smaller. Each step moves the anchor, so holding
        # the hand still sends nothing.
        offset = self.pinch_anchor_y - pinch_y
        while abs(offset) >= VOLUME_STEP:
            if offset > 0:
                self.volume_steps += 1
                self.trigger("VOLUME UP", VK_VOLUME_UP, now)
                self.pinch_anchor_y -= VOLUME_STEP
            else:
                self.volume_steps -= 1
                self.trigger("VOLUME DOWN", VK_VOLUME_DOWN, now)
                self.pinch_anchor_y += VOLUME_STEP
            offset = self.pinch_anchor_y - pinch_y

    # ---- Visuals ---------------------------------------------------------

    def draw_hud(self, img, now, hand_state):
        h, w, _ = img.shape
        self.draw_equalizer(img, now)
        self.draw_trail(img)

        if hand_state is not None:
            hand, pinching = hand_state
            if pinching:
                self.draw_volume_slider(img, hand)
            if self.hold_gesture and not self.hold_fired:
                progress = min((now - self.hold_since) / HOLD_SECONDS, 1.0)
                center = (int(hand[0].x * w), int(hand[0].y * h))
                cv2.ellipse(img, center, (30, 30), -90, 0, int(360 * progress), (0, 220, 255), 4)

        status = "PLAYING" if self.playing else "PAUSED"
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 62), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, f"{status}  vol {self.volume_steps:+d}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "hold fist: play/pause | swipe: track | pinch+move: volume",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        if now - self.action_time < ACTION_FLASH_SECONDS:
            self.draw_action_icon(img, now)

    def draw_equalizer(self, img, now):
        # Decorative bars: they pulse while the music is assumed to be playing
        # and jump with hand movement. They do not analyse the real audio.
        h, w, _ = img.shape
        bars = 24
        bar_w = w // bars
        t = now - self.start_time
        base = 0.35 if self.playing else 0.05
        overlay = img.copy()
        for i in range(bars):
            wave = 0.5 + 0.5 * math.sin(t * 6 + i * 0.7) * math.sin(t * 2.3 + i * 1.3)
            level = min(1.0, base * wave + self.motion_energy * (0.4 + 0.6 * wave))
            bar_h = int(level * h * 0.35) + 4
            hue = int(180 * i / bars)
            hsv = np.uint8([[[hue, 220, 255]]])
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.rectangle(overlay, (i * bar_w + 2, h - bar_h),
                          ((i + 1) * bar_w - 2, h), color, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    def draw_trail(self, img):
        points = list(self.trail)
        for i in range(1, len(points)):
            thickness = 1 + int(8 * i / len(points))
            cv2.line(img, points[i - 1], points[i], (255, 120, 0), thickness)

    def draw_volume_slider(self, img, hand):
        h, w, _ = img.shape
        thumb, index = hand[4], hand[8]
        p1 = (int(thumb.x * w), int(thumb.y * h))
        p2 = (int(index.x * w), int(index.y * h))
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.line(img, p1, p2, (0, 255, 255), 3)
        cv2.circle(img, mid, 10, (0, 255, 255), -1)

        if self.pinch_anchor_y is not None:
            # Tick marks show how far to move for the next volume step.
            x = mid[0] + 40
            anchor = int(self.pinch_anchor_y * h)
            step = int(VOLUME_STEP * h)
            cv2.line(img, (x, anchor - 3 * step), (x, anchor + 3 * step), (180, 180, 180), 2)
            for k in range(-3, 4):
                y = anchor + k * step
                cv2.line(img, (x - 6, y), (x + 6, y), (180, 180, 180), 2)
            cv2.circle(img, (x, mid[1]), 7, (0, 255, 255), -1)
            cv2.putText(img, "+", (x + 12, anchor - 3 * step + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, "-", (x + 12, anchor + 3 * step + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def draw_action_icon(self, img, now):
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2
        # Grow and fade the icon over the flash duration.
        age = (now - self.action_time) / ACTION_FLASH_SECONDS
        size = int(40 + 30 * age)
        overlay = img.copy()
        color = (255, 255, 255)

        if self.action == "PLAY":
            triangle(overlay, (cx - size // 2, cy), size, 1, color)
        elif self.action == "PAUSE":
            bar = size // 3
            cv2.rectangle(overlay, (cx - size // 2, cy - size // 2),
                          (cx - size // 2 + bar, cy + size // 2), color, -1)
            cv2.rectangle(overlay, (cx + size // 2 - bar, cy - size // 2),
                          (cx + size // 2, cy + size // 2), color, -1)
        elif self.action in ("NEXT", "PREVIOUS"):
            direction = 1 if self.action == "NEXT" else -1
            triangle(overlay, (cx - direction * size // 2, cy), size, direction, color)
            triangle(overlay, (cx, cy), size, direction, color)
        else:
            cv2.putText(overlay, "+" if self.action == "VOLUME UP" else "-",
                        (cx - size // 3, cy + size // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 25, color, 6)

        alpha = 0.8 * (1 - age)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, self.action, (cx - 70, cy + size + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def triangle(img, left_mid, size, direction, color):
    """Draw a filled triangle pointing right (direction 1) or left (-1)."""
    x, y = left_mid
    half = size // 2
    if direction > 0:
        points = [(x, y - half), (x, y + half), (x + size, y)]
    else:
        points = [(x + size, y - half), (x + size, y + half), (x, y)]
    cv2.fillPoly(img, [np.array(points, dtype=np.int32)], color)


def main():
    parser = argparse.ArgumentParser(description="Control music with hand gestures.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to a gesture_recognizer.task file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show and print actions without sending media keys.",
    )
    args = parser.parse_args()

    print("Start some music, then control it with your hand. Press q to quit.")
    controller = MusicController(MediaKeys(dry_run=args.dry_run))
    run_camera_app(
        controller.handle_frame,
        model_path=args.model,
        window_size=(640, 480),
        num_hands=1,
    )


if __name__ == "__main__":
    main()
