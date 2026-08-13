"""Internal delay-aligned pose diagnostics and atomic plot publication."""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Union

AngleTriple = tuple[float, float, float]
FrameCoordinate = Union[int, float]


class _AngleDiagnostics:
    """Keep only the source-aligned samples needed by the optional plot."""

    def __init__(self, enabled: bool, max_samples: int) -> None:
        self.enabled = enabled
        self.frames: deque[FrameCoordinate] = deque(maxlen=max_samples)
        self.raw: deque[AngleTriple] = deque(maxlen=max_samples)
        self.smoothed: deque[AngleTriple] = deque(maxlen=max_samples)
        self._break_before_next = False

    def record(
        self,
        frame_index: int,
        raw_pose: AngleTriple | None,
        values: Sequence[float],
    ) -> None:
        if not self.enabled:
            return
        if self._break_before_next and self.frames:
            separator = (self.frames[-1] + frame_index) / 2.0
            missing = (math.nan, math.nan, math.nan)
            self.frames.append(separator)
            self.raw.append(missing)
            self.smoothed.append(missing)
        self._break_before_next = False
        self.frames.append(int(frame_index))
        self.raw.append(raw_pose if raw_pose is not None else (math.nan, math.nan, math.nan))
        self.smoothed.append((float(values[0]), float(values[1]), float(values[2])))

    def save(self, path: str | os.PathLike[str] | None) -> None:
        if path:
            plot_angles(
                list(self.raw),
                list(self.smoothed),
                path,
                frame_indices=list(self.frames),
            )

    def end_segment(self) -> None:
        if self.enabled:
            self._break_before_next = True


def _break_frame_gaps(
    frames: Sequence[FrameCoordinate],
    values: Sequence[AngleTriple],
) -> tuple[list[FrameCoordinate], list[AngleTriple]]:
    """Insert NaN samples so plotting never bridges unobserved frame ranges."""
    broken_frames: list[FrameCoordinate] = []
    broken_values: list[AngleTriple] = []
    previous: FrameCoordinate | None = None
    for frame, value in zip(frames, values):
        if previous is not None and frame > previous + 1:
            broken_frames.append((previous + frame) / 2.0)
            broken_values.append((math.nan, math.nan, math.nan))
        broken_frames.append(frame)
        broken_values.append(value)
        previous = frame
    return broken_frames, broken_values


def plot_angles(
    raw: Sequence[AngleTriple],
    smoothed: Sequence[AngleTriple],
    path: str | os.PathLike[str],
    *,
    frame_indices: Sequence[FrameCoordinate] | None = None,
) -> None:
    """Atomically save source-frame-aligned raw and smoothed pose series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.plot-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    staged = Path(temporary_name)
    fig = None
    try:
        frames = list(range(len(raw))) if frame_indices is None else list(frame_indices)
        raw_frames, raw_values = _break_frame_gaps(frames, raw)
        smoothed_frames, smoothed_values = _break_frame_gaps(frames, smoothed)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        names = ("Yaw", "Pitch", "Roll")
        colors = ("#e74c3c", "#2ecc71", "#2e8fd0")
        for index, (axis, name, color) in enumerate(zip(axes, names, colors)):
            axis.plot(
                raw_frames,
                [value[index] for value in raw_values],
                color="#888888",
                linewidth=0.8,
                alpha=0.6,
                label="raw",
            )
            axis.plot(
                smoothed_frames,
                [value[index] for value in smoothed_values],
                color=color,
                linewidth=1.5,
                label="smoothed",
            )
            axis.set_ylabel(f"{name} (deg)")
            axis.grid(True, alpha=0.3)
        if not frames:
            axes[0].text(
                0.5,
                0.5,
                "No face detections",
                transform=axes[0].transAxes,
                ha="center",
                va="center",
            )
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("frame")
        fig.suptitle("Head pose: raw vs smoothed", fontweight="bold")
        fig.tight_layout()
        fig.savefig(staged, dpi=120)
        os.replace(staged, output)
    finally:
        if fig is not None:
            plt.close(fig)
        staged.unlink(missing_ok=True)
