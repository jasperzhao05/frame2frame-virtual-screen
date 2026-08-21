"""Default backend: MediaPipe landmarks with renderer-aligned head pose."""

from __future__ import annotations

import os

import numpy as np

from ..geometry import rotation_to_euler
from ._canonical_pose import solve_renderer_rotation
from .base import FaceObservation, HeadPose, PoseEstimator


class MediaPipeEstimator(PoseEstimator):
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        yaw_sign: float = 1.0,
        pitch_sign: float = -1.0,
        roll_sign: float = 1.0,
        fps: float = 30.0,
        model_path: str | os.PathLike[str] | None = None,
        focal_length: float | None = None,
    ) -> None:
        from ._facemesh import FaceMeshDetector

        self._detector = FaceMeshDetector(
            model_path=model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            fps=fps,
        )
        self._signs: tuple[float, float, float] = (yaw_sign, pitch_sign, roll_sign)
        self._focal_length = focal_length

    def estimate(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        return self._estimate(frame_bgr, None)

    def estimate_at(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None,
    ) -> FaceObservation | None:
        return self._estimate(frame_bgr, timestamp_ms)

    def _estimate(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None,
    ) -> FaceObservation | None:
        pts = self._detector.process(frame_bgr, timestamp_ms)
        if pts is None:
            return None

        rotation = solve_renderer_rotation(
            pts,
            frame_bgr.shape,
            focal_length=self._focal_length,
        )
        if rotation is None:
            return None
        yaw, pitch, roll = rotation_to_euler(rotation)
        yaw_s, pitch_s, roll_s = self._signs
        pose = HeadPose(yaw=yaw_s * yaw, pitch=pitch_s * pitch, roll=roll_s * roll)

        from ._facemesh import bbox_from_points

        # Landmarks can land slightly outside the frame; keep the box inside it.
        bbox = bbox_from_points(pts, 0, frame_bgr.shape)
        return FaceObservation.from_bbox(pose, bbox, pts)

    def close(self) -> None:
        self._detector.close()
