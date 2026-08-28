"""End-to-end run: decode -> estimate pose -> smooth -> render -> write."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Protocol, Union, cast

import numpy as np

from . import render
from ._diagnostics import _AngleDiagnostics
from ._media import (
    install_video,
    mux_audio,
    resolve_ffmpeg,
    staged_video_path,
)
from ._textures import (
    _PreparedTexture,
    default_texture,
    load_texture,
    prepare_texture,
)
from ._tracking import _PendingFrame, _TemporalTracker
from .config import PipelineConfig
from .content import ContentRequest, ContentSource, LatestFrameSource, VideoContentSource
from .filters import create_filter
from .pose import create_estimator
from .pose.base import FaceObservation
from .video import VideoReader, VideoWriter, WebcamReader

log = logging.getLogger("frame2frame")


class _Reader(Protocol):
    """Frame-source surface used by the orchestration layer."""

    fps: float

    @property
    def size(self) -> tuple[int, int]: ...

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def release(self) -> None: ...


class _Writer(Protocol):
    """Encoded-video sink surface used by the orchestration layer."""

    path: str

    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class _Estimator(Protocol):
    """Minimum caller-injected estimator contract accepted by :func:`run`."""

    def estimate(self, frame_bgr: np.ndarray) -> FaceObservation | None: ...


class _TimestampEstimator(Protocol):
    """Timestamp-aware injected estimator accepted without requiring a base class."""

    def estimate_at(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
    ) -> FaceObservation | None: ...


_EstimatorLike = Union[_Estimator, _TimestampEstimator]


@dataclass
class RunSummary:
    frames: int
    faces: int
    fps: float
    mean_inference_ms: float
    output: str | None
    audio_remuxed: bool = False
    mean_content_ms: float = 0.0
    mean_render_ms: float = 0.0
    content_samples: int = 0
    render_samples: int = 0


@dataclass
class _RunStats:
    frames: int = 0
    faces: int = 0
    inference_seconds: float = 0.0
    inference_samples: int = 0

    @property
    def mean_inference_ms(self) -> float:
        if not self.inference_samples:
            return 0.0
        return float(self.inference_seconds / self.inference_samples * 1000)


class _StaticContentSource:
    """Return one prepared image while sharing the dynamic source path."""

    def __init__(self, content: _PreparedTexture) -> None:
        self.content = content

    def frame_at(self, request: ContentRequest) -> object:
        del request
        return self.content


class _ContentSampler:
    """Resolve a source or callback and normalize its current frame."""

    def __init__(self, source: object) -> None:
        method = getattr(source, "frame_at", None)
        if callable(method):
            self._sample: Callable[[ContentRequest], object | None] = method
        elif callable(source):
            self._sample = cast(Callable[[ContentRequest], Optional[object]], source)
        else:
            raise TypeError("content_source must define frame_at(request) or be callable")

    def sample(self, request: ContentRequest) -> _PreparedTexture | None:
        raw = self._sample(request)
        if raw is None:
            return None
        # Do not cache by object identity. External producers often reuse one
        # mutable grayscale/BGRA buffer; those formats require a fresh color or
        # alpha conversion even when the ndarray object itself is unchanged.
        return prepare_texture(raw)


class _FrameEmitter:
    """Terminal pipeline stage: composite overlays, then write or display."""

    def __init__(
        self,
        cfg: PipelineConfig,
        writer: _Writer | None,
        content: _ContentSampler | None,
    ) -> None:
        self.cfg = cfg
        self.writer = writer
        self.content = content
        self.content_seconds = 0.0
        self.content_samples = 0
        self.render_seconds = 0.0
        self.render_samples = 0

    @property
    def mean_content_ms(self) -> float:
        if not self.content_samples:
            return 0.0
        return self.content_seconds / self.content_samples * 1000.0

    @property
    def mean_render_ms(self) -> float:
        if not self.render_samples:
            return 0.0
        return self.render_seconds / self.render_samples * 1000.0

    def emit(self, packet: _PendingFrame) -> bool:
        frame = packet.frame
        observation = packet.observation
        content = None
        if self.content is not None:
            started = time.perf_counter()
            content = self.content.sample(
                ContentRequest(
                    packet.frame_index,
                    packet.media_time_seconds,
                    self.cfg.webcam is not None,
                )
            )
            self.content_seconds += time.perf_counter() - started
            self.content_samples += 1
        if observation is not None:
            if packet.detected and self.cfg.draw_bbox:
                render.draw_bbox(frame, observation.bbox)
            if packet.detected and self.cfg.draw_axis:
                render.draw_pose_axis(
                    frame,
                    observation.pose.yaw,
                    observation.pose.pitch,
                    observation.pose.roll,
                    observation.center,
                    observation.size,
                )
            if self.cfg.draw_screen and content is not None:
                started = time.perf_counter()
                render.draw_virtual_screen(
                    frame,
                    observation,
                    self.cfg.screen,
                    content,
                    opacity=packet.screen_opacity,
                )
                self.render_seconds += time.perf_counter() - started
                self.render_samples += 1
        if self.writer is not None:
            self.writer.write(frame)  # no-face frames pass through unchanged
        return self.cfg.display and _show(frame)


def _open_reader(cfg: PipelineConfig) -> _Reader:
    if cfg.webcam is not None:
        return WebcamReader(cfg.webcam)
    if not cfg.input:
        raise ValueError("set either input or webcam in the config")
    return VideoReader(cfg.input)


def _load_content(cfg: PipelineConfig) -> _PreparedTexture:
    if cfg.screen.texture_path:
        texture = load_texture(cfg.screen.texture_path)
    else:
        texture = default_texture()
    return prepare_texture(texture)


def _open_content_sampler(
    cfg: PipelineConfig,
    injected: ContentSource | Callable[[ContentRequest], object | None] | None,
    resources: ExitStack,
) -> _ContentSampler | None:
    """Create configured content or borrow a caller-owned injected source."""
    if not cfg.draw_screen:
        return None
    if injected is not None:
        return _ContentSampler(injected)
    if cfg.screen.video_path:
        source = VideoContentSource(cfg.screen.video_path, end_policy=cfg.screen.video_end)
        resources.callback(_close_safely, "screen video", source.close)
        return _ContentSampler(source)
    return _ContentSampler(_StaticContentSource(_load_content(cfg)))


def _close_safely(label: str, callback: Callable[[], object]) -> None:
    """Attempt every cleanup without letting one close mask processing results."""
    try:
        callback()
    except Exception:
        log.exception("failed to close %s", label)


def _estimate_at(
    estimator: _EstimatorLike,
    frame: np.ndarray,
    timestamp_ms: float,
) -> FaceObservation | None:
    estimate_at = getattr(estimator, "estimate_at", None)
    if callable(estimate_at):
        timestamp_estimate = cast(
            Callable[[np.ndarray, float], Optional[FaceObservation]],
            estimate_at,
        )
        return timestamp_estimate(frame, timestamp_ms)
    return cast(_Estimator, estimator).estimate(frame)


def _frame_timestamp(frame_number: int, fps: float, live_origin: float | None) -> float:
    """Return deterministic file time or elapsed monotonic capture time."""
    if live_origin is None:
        return frame_number / fps
    return time.monotonic() - live_origin


def _process_frames(
    reader: _Reader,
    estimator: _EstimatorLike,
    tracking: _TemporalTracker,
    emitter: _FrameEmitter,
    *,
    fps: float,
    live: bool,
) -> _RunStats:
    stats = _RunStats()
    live_origin = time.monotonic() if live else None
    for frame in reader:
        stats.frames += 1
        timestamp_seconds = _frame_timestamp(stats.frames, fps, live_origin)
        started = time.perf_counter()
        observation = _estimate_at(estimator, frame, timestamp_seconds * 1000.0)
        stats.inference_seconds += time.perf_counter() - started
        stats.inference_samples += 1
        if observation is not None:
            stats.faces += 1
        stop_requested = False
        for packet in tracking.push(
            frame,
            observation,
            stats.frames - 1,
            timestamp_seconds,
        ):
            stop_requested = emitter.emit(packet) or stop_requested
        if stop_requested:
            break
    for packet in tracking.finish():
        emitter.emit(packet)
    return stats


def _publish_video(
    cfg: PipelineConfig,
    writer: _Writer | None,
    frames: int,
    fps: float,
    ffmpeg: str | None,
) -> tuple[str | None, bool]:
    if writer is None:
        return None, False
    if not cfg.preserve_audio:
        return install_video(writer.path, cast(str, cfg.output)), False

    output = mux_audio(
        writer.path,
        cast(str, cfg.input),
        cast(str, cfg.output),
        duration_seconds=frames / fps,
        ffmpeg=ffmpeg,
    )
    log.info("remuxed source audio into %s", output)
    return output, True


def run(
    cfg: PipelineConfig,
    estimator: _EstimatorLike | None = None,
    *,
    content_source: ContentSource | Callable[[ContentRequest], np.ndarray | None] | None = None,
) -> RunSummary:
    """Process the configured source.

    Injected estimators and content sources remain caller-owned. Configured
    backends and screen-video sources are created and closed by this function.
    """
    # Validate paths and source selection before opening a camera or truncating
    # an output. FPS-dependent filter limits are checked once the reader exists.
    cfg.validate()
    if content_source is not None and (
        cfg.screen.texture_path is not None or cfg.screen.video_path is not None
    ):
        raise ValueError(
            "injected content_source cannot be combined with a configured screen source"
        )
    if cfg.draw_screen and cfg.webcam is None and isinstance(content_source, LatestFrameSource):
        raise ValueError(
            "LatestFrameSource requires a webcam input; use a timestamp-aware "
            "content source for file processing"
        )
    ffmpeg = resolve_ffmpeg() if cfg.preserve_audio else None
    if cfg.preserve_audio:
        log.warning(
            "audio preservation assumes a constant-frame-rate source; "
            "convert variable-frame-rate input to CFR first to avoid drift"
        )

    diagnostics = _AngleDiagnostics(cfg.plot_path is not None, cfg.max_plot_samples)

    with ExitStack() as resources:
        reader = _open_reader(cfg)
        resources.callback(_close_safely, "video reader", reader.release)
        cfg.validate(reader.fps)
        fps = float(reader.fps)

        if estimator is None:
            # The source frame rate matters to tracking backends; callers can
            # customize other estimator parameters through backend_kwargs.
            backend_kwargs = {"fps": fps, **cfg.backend_kwargs}
            if cfg.backend.strip().lower() in {"mediapipe", "mp", "facemesh"}:
                # Pose fitting and screen projection must use one camera model.
                # ScreenConfig is the single source of truth for that focal length.
                backend_kwargs["focal_length"] = cfg.screen.focal_length
            estimator = create_estimator(cfg.backend, **backend_kwargs)
            resources.callback(_close_safely, "pose estimator", estimator.close)

        smoother = create_filter(fps, cfg.filter)
        content = _open_content_sampler(cfg, content_source, resources)

        # Every file encode is staged. A failed backend, render, writer, plot
        # preparation, or remux therefore leaves an existing output untouched.
        writer_path = None
        if cfg.output:
            writer_path = resources.enter_context(staged_video_path(cfg.output))
        writer = VideoWriter(writer_path, fps, reader.size) if writer_path else None
        if writer is not None:
            resources.callback(_close_safely, "video writer", writer.release)
        if cfg.display:
            resources.callback(_close_safely, "display windows", _destroy_windows)

        emitter = _FrameEmitter(cfg, writer, content)
        tracking = _TemporalTracker(cfg, fps, smoother, diagnostics)
        stats = _process_frames(
            reader,
            estimator,
            tracking,
            emitter,
            fps=fps,
            live=cfg.webcam is not None,
        )

        if writer is not None:
            # ffmpeg must never read a VideoWriter that still has buffered data.
            writer.release()

        # Plot before installing the staged video so a plotting failure also
        # leaves any existing output untouched.
        diagnostics.save(cfg.plot_path)
        output_path, audio_remuxed = _publish_video(cfg, writer, stats.frames, fps, ffmpeg)

    log.info(
        "processed %d frames (%d with a face), mean inference %.1f ms/frame",
        stats.frames,
        stats.faces,
        stats.mean_inference_ms,
    )
    return RunSummary(
        stats.frames,
        stats.faces,
        fps,
        stats.mean_inference_ms,
        output_path,
        audio_remuxed=audio_remuxed,
        mean_content_ms=emitter.mean_content_ms,
        mean_render_ms=emitter.mean_render_ms,
        content_samples=emitter.content_samples,
        render_samples=emitter.render_samples,
    )


def _show(frame: np.ndarray) -> bool:
    import cv2

    cv2.imshow("frame2frame", frame)
    return (cv2.waitKey(1) & 0xFF) == 27  # Esc to quit


def _destroy_windows() -> None:
    import cv2

    cv2.destroyAllWindows()
