"""Shared MediaPipe face helper.

Uses the MediaPipe Tasks FaceLandmarker (the current, non-deprecated API; the
legacy `solutions.FaceMesh` was dropped from the Python 3.13 wheels). It returns
both the 2D landmarks and the head transformation matrix; the default backend
uses the rotation for pose, the deep backends use the landmarks only to crop a
face box. The landmark model is fetched once and cached.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .._downloads import ensure_download

log = logging.getLogger("frame2frame")

_CACHE = Path(
    os.environ.get("FRAME2FRAME_CACHE", Path.home() / ".cache" / "frame2frame")
).expanduser()
_MODEL_NAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
_MODEL_SIZE = 3_758_596


def _resolve_model(path: str | os.PathLike[str] | None) -> Path:
    if path:
        model = Path(path).expanduser()
        if not model.is_file():
            raise FileNotFoundError(f"FaceLandmarker model not found: {model}")
        return model
    return ensure_download(
        _MODEL_URL,
        _CACHE / _MODEL_NAME,
        sha256=_MODEL_SHA256,
        expected_size=_MODEL_SIZE,
    )


class _VideoTimestampSequence:
    """Convert optional source times to MediaPipe's strict integer timeline."""

    def __init__(self, fps: float) -> None:
        if (
            not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("fps must be a positive finite number")
        self._dt_ms: float = 1000.0 / float(fps)
        self._last_source_ms: float | None = None
        self._last_integer_ms: int = -1

    def next(self, source_ms: float | None = None) -> int:
        if source_ms is None:
            source_ms = (
                self._dt_ms if self._last_source_ms is None else self._last_source_ms + self._dt_ms
            )
        if (
            not isinstance(source_ms, (int, float))
            or isinstance(source_ms, bool)
            or not math.isfinite(source_ms)
            or source_ms < 0
        ):
            raise ValueError("video timestamp must be a non-negative finite number")
        source_ms = float(source_ms)
        if self._last_source_ms is not None and source_ms < self._last_source_ms:
            raise ValueError("video timestamps must not move backwards")
        self._last_source_ms = source_ms
        timestamp = max(int(round(source_ms)), self._last_integer_ms + 1)
        self._last_integer_ms = timestamp
        return timestamp


class FaceMeshDetector:
    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        fps: float = 30.0,
    ) -> None:
        if not isinstance(num_faces, int) or isinstance(num_faces, bool) or num_faces < 1:
            raise ValueError("num_faces must be a positive integer")
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("fps must be a positive finite number")

        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self._mp: Any = mp
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_resolve_model(model_path))),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker: Any = FaceLandmarker.create_from_options(options)
        # VIDEO mode wants timestamps at the source cadence; the tracker's
        # motion model assumes them and requires strictly increasing integers.
        self._timestamps = _VideoTimestampSequence(fps)

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(landmarks Nx2 px, head rotation 3x3) of the first face, or (None, None)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._timestamps.next(timestamp_ms))
        if not result.face_landmarks:
            return None, None
        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
        rot = None
        if result.facial_transformation_matrixes:
            rot = np.array(result.facial_transformation_matrixes[0])[:3, :3]
        return pts, rot

    def landmarks(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None = None,
    ) -> np.ndarray | None:
        return self.process(frame_bgr, timestamp_ms)[0]

    def close(self) -> None:
        self._landmarker.close()


def bbox_from_points(
    pts: np.ndarray,
    margin: float,
    frame_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x_min = max(0, int(pts[:, 0].min() - margin))
    y_min = max(0, int(pts[:, 1].min() - margin))
    x_max = min(w - 1, int(pts[:, 0].max() + margin))
    y_max = min(h - 1, int(pts[:, 1].max() + margin))
    return x_min, y_min, x_max, y_max


@dataclass(frozen=True)
class _DetectedFaceCrop:
    """One landmark-backed crop shared by the optional deep estimators."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray


def _detected_face_crop(
    detector: FaceMeshDetector,
    frame_bgr: np.ndarray,
    margin: float,
    timestamp_ms: float | None,
) -> _DetectedFaceCrop | None:
    """Return a valid detected crop without duplicating adapter edge cases."""
    landmarks = detector.landmarks(frame_bgr, timestamp_ms)
    if landmarks is None:
        return None
    bbox = bbox_from_points(landmarks, margin, frame_bgr.shape)
    x_min, y_min, x_max, y_max = bbox
    image = frame_bgr[y_min:y_max, x_min:x_max]
    if image.size == 0:
        return None
    return _DetectedFaceCrop(image, bbox, landmarks)
