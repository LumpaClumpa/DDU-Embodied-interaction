"""Live audio spectrum and "now playing" info for music_control.py.

Both sources run in background threads so the camera loop stays fast. They
are optional: if a library is missing or Windows refuses access, the source
reports why in ``error`` and the rest of the program keeps working.

    AudioSpectrum  captures what the speakers play (WASAPI loopback, via the
                   soundcard package) and turns it into frequency bands.
    NowPlaying     reads title, artist, cover art and play state from the
                   Windows media session (via the winrt packages), the same
                   info shown in the Windows volume flyout.
"""

import asyncio
import ctypes
import sys
import threading
import time
import warnings

import cv2
import numpy as np


SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
FFT_SIZE = 2048
MIN_FREQ = 40
MAX_FREQ = 16000
FLOOR_DB = -70.0        # Bands at or below this level show as empty.
ATTACK = 0.6            # How fast bars rise (0-1)...
DECAY = 0.12            # ...and how fast they fall.

NOW_PLAYING_POLL_SECONDS = 1.0
PLAYBACK_STATUS_PLAYING = 4  # GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING


def _init_com():
    # WASAPI and WinRT need COM initialised in every thread that uses them.
    if sys.platform == "win32":
        ctypes.windll.ole32.CoInitializeEx(None, 0)


class AudioSpectrum:
    def __init__(self, bands=32):
        self.error = ""
        self._samples = np.zeros(FFT_SIZE, dtype=np.float32)
        self._lock = threading.Lock()
        self._levels = np.zeros(bands, dtype=np.float32)
        self._running = True
        self._window = np.hanning(FFT_SIZE).astype(np.float32)

        # Log-spaced band edges mapped to FFT bins, each band at least one bin.
        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        edges = np.geomspace(MIN_FREQ, MAX_FREQ, bands + 1)
        bins = np.searchsorted(freqs, edges)
        self._band_bins = [(lo, max(hi, lo + 1)) for lo, hi in zip(bins[:-1], bins[1:])]

        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        try:
            # soundcard initialises COM itself when imported, and fails if the
            # thread already has COM set up, so import it here in this thread.
            import soundcard as sc

            speaker = sc.default_speaker()
            loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            with warnings.catch_warnings():
                # soundcard warns about gaps whenever nothing is playing.
                warnings.simplefilter("ignore")
                with loopback.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as recorder:
                    while self._running:
                        block = recorder.record(numframes=BLOCK_SIZE)
                        mono = block.mean(axis=1).astype(np.float32)
                        with self._lock:
                            self._samples = np.concatenate((self._samples, mono))[-FFT_SIZE:]
        except ImportError:
            self.error = "audio: install soundcard"
        except Exception as error:  # Any device or driver failure.
            self.error = f"audio: {error}"

    @property
    def available(self):
        return not self.error

    def levels(self):
        """Return smoothed band levels in 0..1, low to high frequencies."""
        with self._lock:
            samples = self._samples.copy()
        spectrum = np.abs(np.fft.rfft(samples * self._window)) / (FFT_SIZE / 4)
        power = np.array([spectrum[lo:hi].max() for lo, hi in self._band_bins])
        db = 20 * np.log10(power + 1e-9)
        target = np.clip((db - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0)

        rising = target > self._levels
        self._levels += np.where(rising, ATTACK, DECAY) * (target - self._levels)
        return self._levels

    def stop(self):
        self._running = False


class NowPlaying:
    def __init__(self):
        self.error = ""
        self.title = ""
        self.artist = ""
        self.cover = None        # BGR image or None.
        self.playing = None      # True / False, or None when unknown.
        self._running = True
        self._cover_key = None
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    @property
    def available(self):
        return not self.error and bool(self.title)

    def _poll(self):
        try:
            _init_com()
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager as SessionManager,
            )
            from winrt.windows.storage.streams import Buffer, InputStreamOptions
        except ImportError:
            self.error = "now playing: install the winrt packages"
            return

        async def read_cover(thumbnail):
            stream = await thumbnail.open_read_async()
            buffer = Buffer(stream.size)
            await stream.read_async(buffer, buffer.capacity, InputStreamOptions.READ_AHEAD)
            data = np.frombuffer(bytes(buffer), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

        async def update():
            manager = await SessionManager.request_async()
            session = manager.get_current_session()
            if session is None:
                self.title, self.artist, self.cover, self.playing = "", "", None, None
                self._cover_key = None
                return
            props = await session.try_get_media_properties_async()
            self.playing = session.get_playback_info().playback_status == PLAYBACK_STATUS_PLAYING
            key = (props.title, props.artist)
            if key != self._cover_key:
                self.cover = await read_cover(props.thumbnail) if props.thumbnail else None
                self._cover_key = key
            self.title, self.artist = props.title or "", props.artist or ""

        while self._running:
            try:
                asyncio.run(update())
                self.error = ""
            except Exception as error:  # The media service can be busy or absent.
                self.error = f"now playing: {error}"
            time.sleep(NOW_PLAYING_POLL_SECONDS)

    def stop(self):
        self._running = False
