"""Backend-agnostic pose estimator interface.

Every backend takes a BGR frame and returns a single FaceObservation (or None
when no face is found). Angles are degrees in the convention the renderer
expects: +yaw turns the gaze toward image-left, +pitch tilts it up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

# Renderers and injected estimators accept subpixel boxes; built-in detectors
# happen to produce integral coordinates. ``int`` remains compatible with float.
BoundingBox = tuple[float, float, float, float]
_EstimatorT = TypeVar("_EstimatorT", bound="PoseEstimator")


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class FaceObservation:
    pose: HeadPose
    center: tuple[float, float]  # face centre in pixels (tdx, tdy)
    size: float  # half the face height in pixels
    bbox: BoundingBox  # x_min, y_min, x_max, y_max
    landmarks: np.ndarray | None = None

    @classmethod
    def from_bbox(
        cls,
        pose: HeadPose,
        bbox: BoundingBox,
        landmarks: np.ndarray | None = None,
    ) -> FaceObservation:
        """Build the derived centre and size shared by every pose backend."""
        x_min, y_min, x_max, y_max = bbox
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        size = max(1.0, (y_max - y_min) / 2.0)
        return cls(pose, center, size, bbox, landmarks)


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> FaceObservation | None: ...

    def estimate_at(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None,
    ) -> FaceObservation | None:
        """Timestamp-aware compatibility hook used by video tracking backends."""
        return self.estimate(frame_bgr)

    def close(self) -> None:
        pass

    def __enter__(self: _EstimatorT) -> _EstimatorT:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
