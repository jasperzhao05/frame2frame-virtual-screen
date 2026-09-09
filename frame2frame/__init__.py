"""frame2frame — a head-relative virtual screen for ordinary footage."""

from __future__ import annotations

from ._version import __version__
from .config import FilterConfig, PipelineConfig, ScreenConfig
from .content import ContentRequest, ContentSource, LatestFrameSource, VideoContentSource
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
    "ContentRequest",
    "ContentSource",
    "LatestFrameSource",
    "VideoContentSource",
    "run",
    "RunSummary",
    "create_estimator",
    "create_filter",
    "HeadPose",
    "FaceObservation",
    "__version__",
]
