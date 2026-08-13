"""Optional backend: 6DRepNet (Hempel et al., 2022).

Predicts head pose through a continuous 6D rotation representation, which avoids
the gimbal-lock failure mode of direct Euler regression. Wraps the third-party
`sixdrepnet` package; treat as experimental until validated on your footage.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import FaceObservation, HeadPose, PoseEstimator


class SixDRepNetEstimator(PoseEstimator):
    def __init__(
        self,
        margin: float = 20,
        fps: float = 30.0,
        face_model_path: str | os.PathLike[str] | None = None,
        **model_kwargs: Any,
    ) -> None:
        try:
            from sixdrepnet import SixDRepNet
        except ImportError as error:
            raise ImportError(
                "the experimental 6DRepNet backend is not installed; its current "
                "PyPI metadata conflicts with MediaPipe's OpenCV provider, so "
                "frame2frame does not offer an automatic extra"
            ) from error

        self._model: Any = SixDRepNet(**model_kwargs)
        self.margin: float = margin

        from ._facemesh import FaceMeshDetector

        self._detector = FaceMeshDetector(model_path=face_model_path, fps=fps)

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
        from ._facemesh import _detected_face_crop

        detected = _detected_face_crop(
            self._detector,
            frame_bgr,
            self.margin,
            timestamp_ms,
        )
        if detected is None:
            return None

        pitch, yaw, roll = self._model.predict(detected.image)
        pose = HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))
        return FaceObservation.from_bbox(pose, detected.bbox, detected.landmarks)

    def close(self) -> None:
        self._detector.close()
