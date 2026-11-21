"""Head-pose maths and the MediaPipe Face Landmarker wrapper.

MediaPipe returns a 4x4 facial transformation matrix that maps the canonical
face model into camera coordinates.  We keep only its rotation block and always
report angles *relative to the calibrated neutral pose*:

    R_rel = R_neutral^T @ R_current

R_rel is then decomposed with a ZYX Euler convention, which for a head is:

    R_rel = Rz(roll) @ Ry(yaw) @ Rx(pitch)

so pitch is a nod about the camera's x axis, yaw a turn about y, roll a tilt
about z.  This is a head-*pose* estimate: it measures how the head is oriented,
not how much each cervical vertebra moved.
"""

from __future__ import annotations

import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(path: Path | str | None = None) -> Path:
    """Return the model path, downloading it once if missing."""
    path = Path(path or config.MODEL_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face_landmarker.task -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def as_rotation(matrix) -> np.ndarray:
    """3x3 rotation block of a 4x4 transform, re-orthonormalised via SVD."""
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    sign = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    return U @ np.diag([1.0, 1.0, sign]) @ Vt


def euler_zyx_deg(R) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees for R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
    R = as_rotation(R)
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def relative_rotation(neutral_R, current_R) -> np.ndarray:
    """Rotation taking the neutral head frame to the current head frame."""
    return as_rotation(neutral_R).T @ as_rotation(current_R)


@dataclass
class PoseResult:
    """One processed frame.

    angles: smoothed (pitch, yaw, roll) in degrees, relative to neutral once
            the tracker has been calibrated, else relative to the camera.
    absolute: the same angles in camera coordinates (debug / sign checking).
    landmarks: the 478 normalised face landmarks, or None.
    stale: True when the face has been lost for longer than the timeout.
    """

    angles: tuple[float, float, float] | None
    absolute: tuple[float, float, float] | None
    landmarks: list | None
    stale: bool


class HeadPoseTracker:
    """Wraps the MediaPipe Face Landmarker and turns matrices into angles."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        smooth_alpha: float | None = None,
        max_jump_deg: float | None = None,
    ) -> None:
        # Imported here so the maths above stays importable without MediaPipe.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(ensure_model(model_path))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._alpha = config.SMOOTHING_ALPHA if smooth_alpha is None else smooth_alpha
        self._max_jump = config.MAX_JUMP_DEG if max_jump_deg is None else max_jump_deg
        self._angles: tuple[float, float, float] | None = None
        self._neutral_R: np.ndarray | None = None
        self._recent_R: deque[np.ndarray] = deque(maxlen=config.CALIBRATION_FRAMES)
        self._last_seen = 0.0
        self._last_ts = 0

    # -- calibration -------------------------------------------------------
    @property
    def has_neutral(self) -> bool:
        return self._neutral_R is not None

    def calibrate(self) -> bool:
        """Average the last few frames and freeze that pose as 0 degrees."""
        if len(self._recent_R) < 3:
            return False
        self._neutral_R = as_rotation(np.mean(np.stack(self._recent_R), axis=0))
        self._angles = None  # restart smoothing in the new reference frame
        return True

    def clear_neutral(self) -> None:
        self._neutral_R = None
        self._angles = None
        self._recent_R.clear()

    # -- per frame ---------------------------------------------------------
    def process(self, bgr: np.ndarray, ts_ms: int | None = None) -> PoseResult:
        ts = int(ts_ms if ts_ms is not None else time.perf_counter() * 1000)
        if ts <= self._last_ts:  # MediaPipe requires strictly increasing stamps
            ts = self._last_ts + 1
        self._last_ts = ts

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            stale = time.perf_counter() - self._last_seen > config.FACE_LOST_TIMEOUT_S
            return PoseResult(None if stale else self._angles, None, None, stale)

        R = as_rotation(result.facial_transformation_matrixes[0])
        self._last_seen = time.perf_counter()
        self._recent_R.append(R)

        reference = self._neutral_R if self._neutral_R is not None else np.eye(3)
        self._angles = self._smooth(euler_zyx_deg(reference.T @ R))
        return PoseResult(
            angles=self._angles,
            absolute=euler_zyx_deg(R),
            landmarks=result.face_landmarks[0],
            stale=False,
        )

    def _smooth(self, raw: tuple[float, float, float]) -> tuple[float, float, float]:
        # shortcut: per-angle EMA rather than a filter in rotation space. Fine at
        # 30 fps with small frame-to-frame deltas; upgrade if you need <1 deg noise.
        prev = self._angles
        if prev is None:
            return raw
        if max(abs(r - p) for r, p in zip(raw, prev)) > self._max_jump:
            return prev  # tracking glitch: hold the last good value
        return tuple(p + self._alpha * (r - p) for p, r in zip(prev, raw))

    def close(self) -> None:
        self._landmarker.close()
