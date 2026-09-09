"""Dynamic visual-content sources for the virtual screen.

The person video remains the master clock. Timestamp-aware sources can select
content from the source packet's presentation time even when temporal tracking
delays emission. ``LatestFrameSource`` deliberately ignores history and returns
the freshest live state instead.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .video import VideoReader

_END_POLICIES = frozenset({"hold", "loop", "hide"})


def _non_negative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _non_negative_time(name: str, value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _validated_frame(frame: object) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("content frame must be a NumPy array")
    if frame.size == 0:
        raise ValueError("content frame must not be empty")
    valid_shape = frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (1, 3, 4))
    if not valid_shape:
        raise ValueError("content frame must be grayscale, BGR, or BGRA")
    return frame


@dataclass(frozen=True)
class ContentRequest:
    """Identify the person-video packet whose screen is about to be rendered.

    ``media_time_seconds`` belongs to the packet, not the moment at which a
    delayed packet happens to leave the filter.  File pipelines should use a
    zero-based media timeline; live pipelines should use monotonic elapsed
    capture time.
    """

    frame_index: int
    media_time_seconds: float
    live: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _non_negative_integer("frame_index", self.frame_index),
        )
        object.__setattr__(
            self,
            "media_time_seconds",
            _non_negative_time("media_time_seconds", self.media_time_seconds),
        )
        if not isinstance(self.live, bool):
            raise ValueError("live must be a boolean")


@runtime_checkable
class ContentSource(Protocol):
    """Narrow caller-adaptation seam consumed by the rendering pipeline.

    Returning ``None`` means that no screen content is available for this
    packet.  The pipeline owns sources it creates from configuration; an
    injected source remains caller-owned.
    """

    def frame_at(self, request: ContentRequest) -> np.ndarray | None: ...


class LatestFrameSource:
    """Thread-safe, bounded source for asynchronously produced content.

    There is deliberately no queue: every publication replaces the previous
    snapshot, so a fast producer cannot build latency or unbounded memory while
    pose estimation is slower.  Safe-copy publication is the default.  With
    ``copy=False`` the source borrows that exact array; the caller must not
    mutate it while the source may still be sampled.

    This object owns no producer thread and therefore needs no close method.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    def publish(self, frame: np.ndarray, *, copy: bool = True) -> None:
        """Replace the current snapshot with ``frame`` in constant storage."""
        if not isinstance(copy, bool):
            raise ValueError("copy must be a boolean")
        image = _validated_frame(frame)
        snapshot = np.array(image, copy=True, order="C") if copy else image
        with self._lock:
            self._latest = snapshot

    def clear(self) -> None:
        """Remove the current snapshot; future samples return ``None``."""
        with self._lock:
            self._latest = None

    def frame_at(self, request: ContentRequest) -> np.ndarray | None:
        """Return the latest snapshot without holding the lock while rendering."""
        if not isinstance(request, ContentRequest):
            raise TypeError("request must be a ContentRequest")
        with self._lock:
            return self._latest


class VideoContentSource:
    """Timestamp-addressed, sequential content decoded with :class:`VideoReader`.

    The selected content index is ``floor(media_time_seconds * fps)``.  Requests
    must be non-decreasing, matching source-order packet emission.  The source
    owns every reader it opens and must be closed by its creator; context-manager
    use is supported for that reason.

    At end of file, ``hold`` retains the final frame, ``hide`` returns ``None``,
    and ``loop`` reopens the video without retaining a decoded-frame cache.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        end_policy: str = "hold",
    ) -> None:
        self.path = self._validated_path(path)
        self.end_policy = self._validated_end_policy(end_policy)
        self._reader: VideoReader | None = None
        self._iterator: Iterator[np.ndarray] | None = None
        self._closed = False
        self._ended = False
        self._last_frame: np.ndarray | None = None
        self._last_request_time: float | None = None
        self._next_absolute_index = 0
        self._frames_in_cycle = 0

        reader = VideoReader(self.path)
        try:
            self.fps = self._validated_fps(reader.fps)
        except BaseException:
            reader.release()
            raise
        self._reader = reader
        self._iterator = iter(reader)

    @staticmethod
    def _validated_path(path: object) -> str:
        if isinstance(path, str) and not path.strip():
            raise ValueError("content video path must be non-empty")
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("content video path must be a string or path-like object")
        return str(Path(path).expanduser())

    @staticmethod
    def _validated_end_policy(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("end_policy must be 'hold', 'loop', or 'hide'")
        normalized = value.strip().lower()
        if normalized not in _END_POLICIES:
            raise ValueError("end_policy must be 'hold', 'loop', or 'hide'")
        return normalized

    @staticmethod
    def _validated_fps(value: object) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("content video fps must be finite and greater than zero")
        return float(value)

    def _target_index(self, timestamp: float) -> int:
        # Advancing by one representable float avoids losing an exact frame
        # boundary to ordinary multiplication round-off (for example 0.3 * 10).
        scaled = math.nextafter(timestamp * self.fps, math.inf)
        return int(math.floor(scaled))

    def _next_frame(self) -> np.ndarray | None:
        if self._iterator is None:
            raise RuntimeError("content video source is closed")
        try:
            frame = next(self._iterator)
        except StopIteration:
            return None
        self._frames_in_cycle += 1
        try:
            return _validated_frame(frame)
        except (TypeError, ValueError) as error:
            raise RuntimeError("content video returned an invalid frame") from error

    def _restart(self) -> None:
        if self._frames_in_cycle == 0:
            raise RuntimeError("content video contains no decodable frames")
        old_reader = self._reader
        self._reader = None
        self._iterator = None
        if old_reader is not None:
            old_reader.release()

        reader = VideoReader(self.path)
        try:
            restarted_fps = self._validated_fps(reader.fps)
            if not math.isclose(restarted_fps, self.fps, rel_tol=1e-12, abs_tol=0.0):
                raise RuntimeError("content video fps changed after looping")
        except BaseException:
            reader.release()
            raise
        self._reader = reader
        self._iterator = iter(reader)
        self._frames_in_cycle = 0

    def frame_at(self, request: ContentRequest) -> np.ndarray | None:
        """Return the content frame presented at ``request.media_time_seconds``."""
        if not isinstance(request, ContentRequest):
            raise TypeError("request must be a ContentRequest")
        if self._closed:
            raise RuntimeError("content video source is closed")
        timestamp = request.media_time_seconds
        if self._last_request_time is not None and timestamp < self._last_request_time:
            raise ValueError("content video timestamps must be non-decreasing")
        self._last_request_time = timestamp

        if self._ended:
            return self._last_frame if self.end_policy == "hold" else None

        target = self._target_index(timestamp)
        while self._next_absolute_index <= target:
            frame = self._next_frame()
            if frame is None:
                if self._frames_in_cycle == 0:
                    raise RuntimeError("content video contains no decodable frames")
                if self.end_policy == "loop":
                    self._restart()
                    continue
                self._ended = True
                return self._last_frame if self.end_policy == "hold" else None
            self._last_frame = frame
            self._next_absolute_index += 1
        return self._last_frame

    def close(self) -> None:
        """Release the owned video reader.  Closing is idempotent."""
        if self._closed:
            return
        self._closed = True
        reader, self._reader = self._reader, None
        self._iterator = None
        if reader is not None:
            reader.release()

    def __enter__(self) -> VideoContentSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "ContentRequest",
    "ContentSource",
    "LatestFrameSource",
    "VideoContentSource",
]
