"""Pose backend registry.

Backends are imported lazily so the core package stays importable with only
numpy/scipy/opencv installed; torch and mediapipe are pulled in only when a
backend that needs them is actually requested.
"""

from __future__ import annotations

from typing import Any

from .base import FaceObservation, HeadPose, PoseEstimator

_ALIASES = {
    "mediapipe": "mediapipe",
    "mp": "mediapipe",
    "facemesh": "mediapipe",
    "hopenet": "hopenet",
    "6drepnet": "sixdrepnet",
    "sixdrepnet": "sixdrepnet",
    "6d": "sixdrepnet",
}


def available_backends() -> tuple[str, ...]:
    """Return built-in backend names; optional dependencies may still be absent."""
    return ("mediapipe", "hopenet", "6drepnet")


def create_estimator(name: str = "mediapipe", **kwargs: Any) -> PoseEstimator:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"pose backend name must be one of {available_backends()}")
    key = _ALIASES.get(name.strip().lower())
    if key == "mediapipe":
        from .mediapipe_face import MediaPipeEstimator

        return MediaPipeEstimator(**kwargs)
    if key == "hopenet":
        from .hopenet import HopenetEstimator

        return HopenetEstimator(**kwargs)
    if key == "sixdrepnet":
        from .sixdrepnet import SixDRepNetEstimator

        return SixDRepNetEstimator(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}; choose from {available_backends()}")


__all__ = [
    "PoseEstimator",
    "HeadPose",
    "FaceObservation",
    "create_estimator",
    "available_backends",
]
