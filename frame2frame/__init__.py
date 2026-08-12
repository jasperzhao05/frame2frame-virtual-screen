"""frame2frame — a head-locked virtual screen driven by smoothed head pose."""

from __future__ import annotations

from ._version import __version__
from .config import FilterConfig, PipelineConfig, ScreenConfig
from .filters import create_filter
from .pipeline import RunSummary, run
from .pose import (
    FaceObservation,
    HeadPose,
    create_estimator,
)

__all__ = [
    "PipelineConfig",
    "ScreenConfig",
    "FilterConfig",
    "run",
    "RunSummary",
    "create_estimator",
    "create_filter",
    "HeadPose",
    "FaceObservation",
    "__version__",
]
