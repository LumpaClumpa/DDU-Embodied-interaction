"""Iteration 2: control music playback with your hand and see it on screen.

Iteration 1 reacts to the gesture class from the model. This iteration reads
the hand landmarks (finger positions and hand movement) directly instead:

    Hold a closed fist             -> play / pause
    Swipe open hand right / left   -> next / previous track
    Pinch thumb + index, move up   -> volume up
    Pinch thumb + index, move down -> volume down
    Frame with both hands          -> open / resize the visualizer
    (thumb + index of each hand as an "L", drag apart, then relax the hands;
     a tiny frame closes the visualizer)

Gestures from the trained model add more commands once they are trained
(the dataset folder names must match the labels in TRAINED_GESTURES):

    Hold palm_stop                 -> pause
    Hold point_right / point_left  -> next / previous track
    Hold thumb_down                -> mute / unmute
    Hold rock_horns                -> rock mode colours on / off
    Hold shaka                     -> shuffle jump (skip 2-5 tracks)

Commands are sent as Windows media keys, so they control whatever player is
active (Spotify, YouTube in a browser, Windows Media Player, ...). The only
interface on screen is the swipe meter, plus the visualizer when it is open.

Run:
    python music_control.py
    python music_control.py --dry-run   (show actions without sending keys)
"""

import argparse
import ctypes
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from gesture_camera import (
    MODEL_PATH,
    draw_text,
    run_camera_app,
    text_size,
    top_gesture,
)
from media_sources import AudioSpectrum, NowPlaying


HOLD_SECONDS = 0.2      # How long a gesture must be held before it triggers.
MIN_SCORE = 0.4         # Ignore uncertain model predictions.

# Labels (lowercase dataset folder names) of trained gestures this iteration
# reacts to. Labels the model does not know are simply never seen.
TRAINED_GESTURES = {
    "palm_stop",
    "point_left",
    "point_right",
    "rock_horns",
    "shaka",
    "thumb_down",
}

SWIPE_DISTANCE = 0.25      # Fraction of frame width the hand must travel...
SWIPE_WINDOW = 0.40        # ...within this many seconds.
SWIPE_MIN_SPEED = 0.8      # Frame widths per second while swiping.
SWIPE_STRAIGHTNESS = 2.0   # Sideways travel must be this many times the vertical.
SWIPE_SMOOTHNESS = 1.3     # Path length may be at most this times the net travel.
SWIPE_MIN_FRAMES = 4       # A swipe spans at least this many frames...
SWIPE_MAX_STEP_SHARE = 0.5 # ...and no single frame may cover more than this
                           # share of it, so a tracking glitch is not a swipe.
SWIPE_SETTLE = 0.3        # Seconds an open hand must be visible before swiping.
SWIPE_COOLDOWN = 0.6       # Minimum time between two swipes.
SWIPE_RETURN_BLOCK = 0.9   # Opposite direction is ignored this long (return stroke).

PINCH_RATIO = 0.35      # Pinch when tip distance < this * hand size.
VOLUME_STEP = 0.04      # Fraction of frame height per volume key press.

FRAME_HOLD = 0.3        # Both hands must show the "L" this long to start framing.
FRAME_RELEASE = 0.25    # Framing ends after the "L" is gone this long.
FRAME_MIN_SIZE = 0.15   # Frames narrower or lower than this (share of the
                        # image) close the visualizer instead of placing it.

VISUALIZER_BANDS = 32
TEXT_COLOR = (0, 0, 0)          # Black text...
PANEL_COLOR = (245, 245, 245)   # ...on light panels, so it reads on any video.
BAR_COLOR = (60, 60, 60)
ROCK_BAR_COLORS = ((0, 60, 230), (0, 200, 255))  # Red to yellow (BGR).

VK_VOLUME_MUTE = 0xAD
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


def extended_fingers(hand):
    """Number of fingers (not the thumb) whose tip is farther from the wrist
    than its middle joint."""
    wrist = hand[0]
    return sum(
        distance(wrist, hand[tip]) > distance(wrist, hand[joint])
        for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18))
    )


def is_fist(hand):
    """True when all four fingertips are curled closer to the wrist than
    their middle joints (tips 8/12/16/20, joints 6/10/14/18) and the thumb
    is tucked in, so a thumbs down is not mistaken for a fist."""
    wrist = hand[0]
    fingers_curled = all(
        distance(wrist, hand[tip]) < distance(wrist, hand[joint])
        for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18))
    )
    hand_size = max(distance(wrist, hand[9]), 1e-6)
    thumb_tucked = distance(hand[4], hand[9]) < 0.7 * hand_size
    return fingers_curled and thumb_tucked


def is_frame_hand(hand):
    """True for the "L" of a camera-frame gesture: index finger out, thumb
    spread away from it, middle and ring fingers curled."""
    wrist = hand[0]
    hand_size = max(distance(wrist, hand[9]), 1e-6)
    index_out = distance(wrist, hand[8]) > distance(wrist, hand[6])
    middle_in = distance(wrist, hand[12]) < distance(wrist, hand[10])
    ring_in = distance(wrist, hand[16]) < distance(wrist, hand[14])
    thumb_out = distance(hand[4], hand[5]) > 0.6 * hand_size
    return index_out and middle_in and ring_in and thumb_out


def frame_corner(hand):
    """Inner corner of the "L": between the thumb and index finger bases."""
    return ((hand[2].x + hand[5].x) / 2, (hand[2].y + hand[5].y) / 2)


class MusicController:
    def __init__(self, keys, audio=None, now_playing=None):
        self.keys = keys
        self.audio = audio
        self.now_playing = now_playing
        self.playing = False  # Assumed state, unless the media session reports it.
        self.muted = False     # Assumed state, toggled by thumb_down.
        self.rock_mode = False
        self.volume_steps = 0  # Net volume steps sent since start.

        self.hold_gesture = ""
        self.hold_since = 0.0
        self.hold_fired = False

        self.hand_since = None      # When the open hand appeared.
        self.swipe_history = deque()  # (time, x, y) of the palm centre.
        self.last_swipe = 0.0
        self.last_swipe_dir = 0
        self.swipe_meter = 0.0      # -1..1, share of SWIPE_DISTANCE travelled.
        self.swipe_ready = False    # True when the current move would count.

        self.pinch_anchor_y = None

        # Visualizer frame, as (x0, y0, x1, y1) in 0..1 image coordinates.
        self.viz_rect = None
        self.frame_rect = None      # Rectangle being dragged out right now.
        self.frame_seen = None      # Last time both hands showed the "L".
        self.frame_since = None     # When the current "L" pose started.

    def trigger(self, action, vk_code, now, presses=1):
        if vk_code is None:
            # Visual-only action: nothing is sent to the music player.
            print(f"[{time.strftime('%H:%M:%S')}] {action}")
        else:
            for _ in range(presses):
                self.keys.press(vk_code, action)

    def handle_frame(self, img, results):
        now = time.monotonic()
        if self.now_playing is not None and self.now_playing.playing is not None:
            self.playing = self.now_playing.playing

        hands = results.hand_landmarks
        framing = self.update_frame(hands, now)
        if framing or len(hands) != 1:
            # No controls while framing, or while two hands make it unclear
            # which one is in charge.
            self.reset_controls()
        else:
            self.update_controls(hands[0], results, now)

        self.draw(img, now, show_meter=len(hands) == 1 and not framing)

    def reset_controls(self):
        self.reset_swipe()
        self.hand_since = None
        self.pinch_anchor_y = None
        self.hold_gesture = ""

    def update_controls(self, hand, results, now):
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

        self.update_hold(self.held_gesture(results, fist), now)
        if pinching:
            self.update_volume(thumb_tip, index_tip, now)
            self.reset_swipe()
            self.hand_since = None
        else:
            self.pinch_anchor_y = None
            self.update_swipe(hand, now)

    def update_frame(self, hands, now):
        """Track the two-hand frame gesture. Returns True while framing."""
        if len(hands) == 2 and all(is_frame_hand(hand) for hand in hands):
            if self.frame_since is None:
                self.frame_since = now
            self.frame_seen = now
            if now - self.frame_since >= FRAME_HOLD:
                (ax, ay), (bx, by) = (frame_corner(hand) for hand in hands)
                # Landmarks can lie slightly outside the image; keep the frame inside.
                self.frame_rect = tuple(
                    min(max(value, 0.0), 1.0)
                    for value in (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                )
            return self.frame_rect is not None

        self.frame_since = None
        if self.frame_rect is None:
            return False
        if now - self.frame_seen < FRAME_RELEASE:
            return True  # Brief tracking loss: keep the frame as it is.

        # The hands relaxed: place the visualizer, or close it if tiny.
        x0, y0, x1, y1 = self.frame_rect
        if x1 - x0 < FRAME_MIN_SIZE or y1 - y0 < FRAME_MIN_SIZE:
            self.viz_rect = None
            self.trigger("VISUALIZER CLOSED", None, now)
        else:
            self.viz_rect = self.frame_rect
            self.trigger("VISUALIZER PLACED", None, now)
        self.frame_rect = None
        return False

    @staticmethod
    def held_gesture(results, fist):
        """Name of the gesture to hold: a trained gesture wins over the fist."""
        gesture = top_gesture(results, 0)
        if gesture is not None and gesture.score >= MIN_SCORE:
            name = gesture.category_name.lower()
            if name in TRAINED_GESTURES:
                return name
        return "fist" if fist else ""

    def update_hold(self, name, now):
        if name != self.hold_gesture:
            self.hold_gesture = name
            self.hold_since = now
            self.hold_fired = False
        elif name and not self.hold_fired and now - self.hold_since >= HOLD_SECONDS:
            self.hold_fired = True
            self.run_gesture_action(name, now)

    def run_gesture_action(self, name, now):
        if name == "fist":
            self.playing = not self.playing
            self.trigger("PLAY" if self.playing else "PAUSE", VK_MEDIA_PLAY_PAUSE, now)
        elif name == "palm_stop":
            # There is no reliable "pause only" media key, so the toggle is
            # sent only when the music is playing.
            if self.playing:
                self.playing = False
                self.trigger("PAUSE", VK_MEDIA_PLAY_PAUSE, now)
            else:
                self.trigger("ALREADY PAUSED", None, now)
        elif name == "point_right":
            self.trigger("NEXT", VK_MEDIA_NEXT_TRACK, now)
        elif name == "point_left":
            self.trigger("PREVIOUS", VK_MEDIA_PREV_TRACK, now)
        elif name == "thumb_down":
            self.muted = not self.muted
            self.trigger("MUTE" if self.muted else "UNMUTE", VK_VOLUME_MUTE, now)
        elif name == "rock_horns":
            self.rock_mode = not self.rock_mode
            self.trigger("ROCK MODE " + ("ON" if self.rock_mode else "OFF"), None, now)
        elif name == "shaka":
            # Media keys have no shuffle, so skip a random number of tracks.
            skips = random.randint(2, 5)
            self.trigger(f"SHUFFLE +{skips}", VK_MEDIA_NEXT_TRACK, now, presses=skips)

    def reset_swipe(self):
        self.swipe_history.clear()
        self.swipe_meter = 0.0
        self.swipe_ready = False

    def update_swipe(self, hand, now):
        # Only an open hand (at least 3 fingers out) can swipe, and only after
        # it has been visible for a moment, so moving a relaxed hand around or
        # bringing it into view does not count.
        if extended_fingers(hand) < 3:
            self.reset_swipe()
            self.hand_since = None
            return
        if self.hand_since is None:
            self.hand_since = now
        if now - self.hand_since < SWIPE_SETTLE:
            self.reset_swipe()
            return

        palm = hand[9]  # Middle-finger knuckle: steadier than the fingertips.
        self.swipe_history.append((now, palm.x, palm.y))
        while self.swipe_history and now - self.swipe_history[0][0] > SWIPE_WINDOW:
            self.swipe_history.popleft()

        # Try every start point in the window: the swipe may have started
        # after the oldest sample. Keep the longest straight, smooth segment,
        # preferring segments that were also fast enough.
        points = list(self.swipe_history)
        best_dx, best_fast = 0.0, False
        for i in range(len(points) - SWIPE_MIN_FRAMES + 1):
            t0, x0, y0 = points[i]
            dx, dy = palm.x - x0, palm.y - y0
            steps = [abs(points[k + 1][1] - points[k][1]) for k in range(i, len(points) - 1)]
            if (
                abs(dx) < SWIPE_STRAIGHTNESS * abs(dy)
                or sum(steps) > SWIPE_SMOOTHNESS * abs(dx)
                or max(steps) > SWIPE_MAX_STEP_SHARE * abs(dx)
            ):
                continue
            fast = abs(dx) / max(now - t0, 1e-3) >= SWIPE_MIN_SPEED
            if (fast, abs(dx)) > (best_fast, abs(best_dx)):
                best_dx, best_fast = dx, fast

        direction = 1 if best_dx > 0 else -1
        since_last = now - self.last_swipe
        blocked = since_last < SWIPE_COOLDOWN or (
            direction == -self.last_swipe_dir and since_last < SWIPE_RETURN_BLOCK
        )
        self.swipe_meter = max(-1.0, min(1.0, best_dx / SWIPE_DISTANCE))
        self.swipe_ready = best_fast and not blocked

        if self.swipe_ready and abs(best_dx) >= SWIPE_DISTANCE:
            self.last_swipe = now
            self.last_swipe_dir = direction
            self.swipe_history.clear()
            if direction > 0:
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

    def draw(self, img, now, show_meter):
        if self.viz_rect is not None and self.frame_rect is None:
            self.draw_visualizer(img, self.viz_rect)
        if self.frame_rect is not None:
            self.draw_frame_outline(img, self.frame_rect)
        if show_meter:
            self.draw_swipe_meter(img, now)

    def draw_swipe_meter(self, img, now):
        """Horizontal bar at the top: fills left or right as the hand
        travels, green when the move is fast and straight enough to count."""
        h, w, _ = img.shape
        cx, y, half = w // 2, 24, 140
        cv2.rectangle(img, (cx - half, y - 8), (cx + half, y + 8), PANEL_COLOR, -1)
        cv2.line(img, (cx, y - 12), (cx, y + 12), TEXT_COLOR, 2)

        just_swiped = now - self.last_swipe < 0.3
        fill = self.last_swipe_dir if just_swiped else self.swipe_meter
        if fill:
            color = (0, 220, 0) if self.swipe_ready or just_swiped else (0, 160, 255)
            end = cx + int(fill * half)
            cv2.rectangle(img, (min(cx, end), y - 8), (max(cx, end), y + 8), color, -1)

        prev_w, label_h = text_size("PREV", 13, bold=True)
        draw_text(img, "PREV", (cx - half - prev_w - 12, y - label_h // 2 - 1), 13,
                  TEXT_COLOR, bold=True, background=PANEL_COLOR)
        draw_text(img, "NEXT", (cx + half + 12, y - label_h // 2 - 1), 13,
                  TEXT_COLOR, bold=True, background=PANEL_COLOR)

    @staticmethod
    def draw_frame_outline(img, rect):
        """Corner brackets while framing: white when the frame will place the
        visualizer, red when it is small enough to close it."""
        h, w, _ = img.shape
        x0, y0, x1, y1 = rect
        closes = x1 - x0 < FRAME_MIN_SIZE or y1 - y0 < FRAME_MIN_SIZE
        color = (40, 40, 230) if closes else (255, 255, 255)
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        arm = max(12, min(p1[0] - p0[0], p1[1] - p0[1]) // 4)
        for (x, y), (sx, sy) in (
            ((p0[0], p0[1]), (1, 1)), ((p1[0], p0[1]), (-1, 1)),
            ((p0[0], p1[1]), (1, -1)), ((p1[0], p1[1]), (-1, -1)),
        ):
            cv2.line(img, (x, y), (x + sx * arm, y), color, 4)
            cv2.line(img, (x, y), (x, y + sy * arm), color, 4)
        cv2.rectangle(img, p0, p1, color, 1)

    def draw_visualizer(self, img, rect):
        h, w, _ = img.shape
        x0, y0 = int(rect[0] * w), int(rect[1] * h)
        x1, y1 = int(rect[2] * w), int(rect[3] * h)
        pad = 10

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # Now playing: cover art on the left, title and artist beside it.
        bars_top = y0 + pad
        info = self.now_playing
        if info is not None and info.available:
            art = min(80, (y1 - y0) // 2, (x1 - x0) // 3)
            text_x = x0 + pad
            if info.cover is not None and art >= 24:
                cover = cv2.resize(info.cover, (art, art), interpolation=cv2.INTER_AREA)
                img[y0 + pad:y0 + pad + art, x0 + pad:x0 + pad + art] = cover
                text_x += art + pad
            max_chars = max(4, (x1 - text_x - pad) // 9)
            draw_text(img, shorten(info.title, max_chars), (text_x, y0 + pad + 2), 16,
                      TEXT_COLOR, bold=True)
            draw_text(img, shorten(info.artist, max_chars + 4), (text_x, y0 + pad + 24), 13,
                      TEXT_COLOR)
            bars_top = y0 + pad + max(art, 44) + pad

        # Spectrum bars along the bottom of the frame.
        if self.audio is None or not self.audio.available:
            message = self.audio.error if self.audio is not None else "audio: off"
            draw_text(img, message, (x0 + pad, y1 - pad - 16), 13, TEXT_COLOR)
            return
        levels = self.audio.levels()
        area_w, area_h = x1 - x0 - 2 * pad, y1 - pad - bars_top
        if area_w < len(levels) or area_h < 8:
            return
        bar_w = area_w / len(levels)
        for i, level in enumerate(levels):
            bar_h = max(2, int(level * area_h))
            bx0 = int(x0 + pad + i * bar_w)
            bx1 = int(x0 + pad + (i + 1) * bar_w) - 2
            cv2.rectangle(img, (bx0, y1 - pad - bar_h), (bx1, y1 - pad),
                          self.bar_color(i / len(levels)), -1)

    def bar_color(self, position):
        if not self.rock_mode:
            return BAR_COLOR
        low, high = ROCK_BAR_COLORS
        return tuple(int(a + (b - a) * position) for a, b in zip(low, high))


def shorten(text, max_chars):
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


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
    audio, now_playing = AudioSpectrum(VISUALIZER_BANDS), NowPlaying()
    controller = MusicController(MediaKeys(dry_run=args.dry_run), audio, now_playing)
    try:
        run_camera_app(
            controller.handle_frame,
            model_path=args.model,
            window_size=(640, 480),
            num_hands=2,
        )
    finally:
        audio.stop()
        now_playing.stop()
        for source in (audio, now_playing):
            if source.error:
                print(source.error)


if __name__ == "__main__":
    main()
