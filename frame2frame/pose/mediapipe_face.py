"""Default backend: MediaPipe FaceLandmarker head pose.

Reads the head transformation matrix straight from FaceLandmarker and converts
it to yaw/pitch/roll. This is far steadier than regressing Euler angles out of a
solvePnP fit, which suffered from a 180 degree offset and sign flips near
profile views. MediaPipe reports rotation in a y-up / z-out-of-screen frame, so
we change basis into the renderer's OpenCV camera frame before decomposing.
"""

from __future__ import annotations

import os

import numpy as np

from ..geometry import rotation_to_euler
from .base import FaceObservation, HeadPose, PoseEstimator

# mediapipe (x-right, y-up, z-toward-viewer) -> OpenCV camera (x-right, y-down, z-in)
_BASIS = np.diag([1.0, -1.0, -1.0])


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
    ) -> None:
        from ._facemesh import FaceMeshDetector

        self._detector = FaceMeshDetector(
            model_path=model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            fps=fps,
        )
        self._signs: tuple[float, float, float] = (yaw_sign, pitch_sign, roll_sign)

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
        pts, rot = self._detector.process(frame_bgr, timestamp_ms)
        if pts is None or rot is None:
            return None

        yaw, pitch, roll = rotation_to_euler(_BASIS @ rot @ _BASIS)
        yaw_s, pitch_s, roll_s = self._signs
        pose = HeadPose(yaw=yaw_s * yaw, pitch=pitch_s * pitch, roll=roll_s * roll)

        from ._facemesh import bbox_from_points

        # Landmarks can land slightly outside the frame; keep the box inside it.
        bbox = bbox_from_points(pts, 0, frame_bgr.shape)
        return FaceObservation.from_bbox(pose, bbox, pts)

    def close(self) -> None:
        self._detector.close()
