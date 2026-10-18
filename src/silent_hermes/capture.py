"""Webcam capture. Headless: no preview window, frames stay in memory.

Only mouth crops come out of here (via roi.mouth_crops); raw frames are dropped.
"""
from __future__ import annotations

import select
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .roi import FaceKeypoints, mouth_crops, resample


@dataclass
class Clip:
    crops: np.ndarray | None  # (T, 96, 96) uint8 at 25 fps
    face_rate: float          # fraction of frames with a detected face
    seconds: float


class Camera:
    def __init__(self, device: int | str = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if isinstance(device, int) else cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):  # let auto-exposure settle
            self.cap.read()

    def read_rgb(self) -> np.ndarray | None:
        ok, f = self.cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None

    def record(self, stop: threading.Event | None = None, seconds: float | None = None) -> tuple[list, float]:
        """Grab frames until `stop` is set or `seconds` pass. Returns (frames, measured fps)."""
        frames, t0 = [], time.monotonic()
        while True:
            f = self.read_rgb()
            if f is not None:
                frames.append(f)
            el = time.monotonic() - t0
            if (seconds is not None and el >= seconds) or (stop is not None and stop.is_set()):
                break
        el = time.monotonic() - t0
        return frames, (len(frames) / el if el > 0 else 25.0)

    def close(self):
        self.cap.release()


def to_clip(frames: list, fps: float, detector: FaceKeypoints) -> Clip:
    secs = len(frames) / fps if fps else 0.0
    frames = resample(frames, fps)
    lms = [detector(f) for f in frames]
    rate = sum(l is not None for l in lms) / max(1, len(lms))
    return Clip(mouth_crops(frames, lms), rate, secs)


def record_timed(cam: Camera, det: FaceKeypoints, seconds: float) -> Clip:
    frames, fps = cam.record(seconds=seconds)
    return to_clip(frames, fps, det)


def record_until_enter(cam: Camera, det: FaceKeypoints, max_seconds: float = 8.0) -> Clip:
    """Start immediately; stop when Enter is pressed (or at max_seconds)."""
    stop = threading.Event()

    def wait_enter():
        end = time.monotonic() + max_seconds
        while not stop.is_set() and time.monotonic() < end:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                sys.stdin.readline()
                break
        stop.set()

    th = threading.Thread(target=wait_enter, daemon=True)
    th.start()
    frames, fps = cam.record(stop=stop)
    th.join()
    return to_clip(frames, fps, det)
