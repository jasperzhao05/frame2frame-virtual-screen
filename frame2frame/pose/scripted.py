"""A pose estimator that replays a known sequence instead of detecting a face.

Useful for demos and tests where you want to exercise the smoothing and
rendering path without depending on mediapipe or real footage.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import FaceObservation, PoseEstimator


class ScriptedEstimator(PoseEstimator):
    def __init__(
        self,
        fn: Callable[[int, np.ndarray], FaceObservation | None],
    ) -> None:
        self._fn = fn
        self._i: int = 0

    def estimate(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        observation = self._fn(self._i, frame_bgr)
        self._i += 1
        return observation
