"""Face keypoints -> aligned 96x96 grey mouth crops (Auto-AVSR preprocessing).

Ported from Auto-AVSR's mediapipe VideoProcess, moved to the MediaPipe Tasks API
(FaceDetector; mp.solutions is gone in mediapipe >= 0.10.30). Only the mouth crop
leaves this module; full frames are never stored.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .vsr import ASSETS

CROP = 96
_WINDOW = 12  # landmark smoothing window (frames), as upstream


class FaceKeypoints:
    """BlazeFace short-range: right eye, left eye, nose tip, mouth centre (pixels)."""

    def __init__(self, model_path: Path, min_conf: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_conf,
        )
        self.det = vision.FaceDetector.create_from_options(opts)

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        import mediapipe as mp

        res = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)))
        if not res.detections:
            return None
        h, w = frame_rgb.shape[:2]
        best = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        return np.array([[k.x * w, k.y * h] for k in best.keypoints[:4]], dtype=np.float32)

    def close(self):
        self.det.close()


def _stable_reference() -> np.ndarray:
    ref = np.load(ASSETS / "20words_mean_face.npy")
    return np.vstack([ref[36:42].mean(0), ref[42:48].mean(0), ref[31:36].mean(0), ref[48:68].mean(0)])


_REF = None


def interpolate(landmarks: list[np.ndarray | None]) -> list[np.ndarray] | None:
    idx = [i for i, l in enumerate(landmarks) if l is not None]
    if not idx:
        return None
    out = list(landmarks)
    for a, b in zip(idx, idx[1:]):
        for k in range(1, b - a):
            out[a + k] = out[a] + k / (b - a) * (out[b] - out[a])
    out[: idx[0]] = [out[idx[0]]] * idx[0]
    out[idx[-1]:] = [out[idx[-1]]] * (len(out) - idx[-1])
    return out


def mouth_crops(frames_rgb: list[np.ndarray], landmarks: list[np.ndarray | None]) -> np.ndarray | None:
    """Align each frame to the mean face and cut a 96x96 grey patch at the mouth."""
    global _REF
    if _REF is None:
        _REF = _stable_reference()
    lms = interpolate(landmarks)
    if lms is None:
        return None
    n, out = len(frames_rgb), []
    for i, frame in enumerate(frames_rgb):
        m = min(_WINDOW // 2, i, n - 1 - i)
        smooth = np.mean(lms[i - m: i + m + 1], axis=0)
        smooth += lms[i].mean(0) - smooth.mean(0)
        tf = cv2.estimateAffinePartial2D(smooth, _REF, method=cv2.LMEDS)[0]
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        warped = cv2.warpAffine(grey, tf, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mouth = smooth[3] @ tf[:, :2].T + tf[:, 2]
        cx, cy = mouth
        y0 = int(round(np.clip(cy - CROP // 2, 0, 256 - CROP)))
        x0 = int(round(np.clip(cx - CROP // 2, 0, 256 - CROP)))
        out.append(warped[y0: y0 + CROP, x0: x0 + CROP])
    return np.stack(out).astype(np.uint8)


def resample(frames: list, src_fps: float, dst_fps: float = 25.0) -> list:
    """Nearest-frame resample to the model's 25 fps."""
    if abs(src_fps - dst_fps) < 0.5 or not frames:
        return frames
    n = max(1, int(round(len(frames) * dst_fps / src_fps)))
    return [frames[min(len(frames) - 1, int(round(i * src_fps / dst_fps)))] for i in range(n)]


def crops_from_video(path: Path, detector: FaceKeypoints) -> tuple[np.ndarray | None, float]:
    """Read a video file -> (mouth crops at 25 fps, fraction of frames with a face)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = resample(frames, fps)
    lms = [detector(f) for f in frames]
    hit = sum(l is not None for l in lms) / max(1, len(lms))
    return mouth_crops(frames, lms), hit
