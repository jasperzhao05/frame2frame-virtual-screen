"""Internal temporal tracking, dropout recovery, and delay alignment."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._diagnostics import AngleTriple, _AngleDiagnostics
from .config import PipelineConfig
from .filters import _PoseFilter
from .pose.base import BoundingBox, FaceObservation, HeadPose

log = logging.getLogger("frame2frame")


@dataclass(frozen=True)
class _PoseSample:
    """One validated pose and face-translation sample."""

    yaw: float
    pitch: float
    roll: float
    center_x: float
    center_y: float
    face_size: float

    @property
    def attitude(self) -> AngleTriple:
        return self.yaw, self.pitch, self.roll

    @property
    def translation(self) -> tuple[float, float, float]:
        return self.center_x, self.center_y, self.face_size


@dataclass(frozen=True)
class _ObservationSnapshot:
    """Detection data retained while brief dropouts hold the last pose."""

    sample: _PoseSample
    bbox: BoundingBox
    landmarks: np.ndarray | None

    def to_observation(self) -> FaceObservation:
        sample = self.sample
        return FaceObservation(
            HeadPose(*sample.attitude),
            (sample.center_x, sample.center_y),
            sample.face_size,
            self.bbox,
            self.landmarks,
        )


@dataclass
class _PendingFrame:
    frame: np.ndarray
    observation: FaceObservation | None
    raw_pose: AngleTriple | None = None
    frame_index: int = 0
    detected: bool = False
    screen_opacity: float = 0.0


class _TemporalTracker:
    """Own filter progression, dropout segments, and source-order alignment.

    Once a face starts a tracking segment, every decoded frame advances the
    filter. Brief detection gaps hold the last pose; a sustained gap flushes
    the delayed tail and resets the filter. One queue contains unresolved
    frames in source order and releases them as their filtered poses mature.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        fps: float,
        smoother: _PoseFilter,
        diagnostics: _AngleDiagnostics,
    ) -> None:
        self.cfg = cfg
        self.smoother = smoother
        self.diagnostics = diagnostics
        self.fps = float(fps)
        self.latency = self._compensation_latency()

        self.delayed: deque[_PendingFrame] = deque()
        self.anchor: _ObservationSnapshot | None = None
        self.last_timestamp_seconds: float | None = None
        self.last_detection_timestamp_seconds: float | None = None
        self.misses = 0

        if self.latency:
            log.info("compensating FIR group delay: %d video frames", self.latency)

    def push(
        self,
        frame: np.ndarray,
        observation: FaceObservation | None,
        frame_index: int = 0,
        timestamp_seconds: float | None = None,
    ) -> list[_PendingFrame]:
        """Accept one decoded frame and return packets ready for output."""
        timestamp_seconds = self._validated_timestamp(timestamp_seconds)
        self._check_timestamp_order(timestamp_seconds)

        detected = observation is not None
        ready = self._close_stale_segment_before_reacquisition(
            detected,
            timestamp_seconds,
        )
        packet, dropout_elapsed = self._packet_for_sample(
            frame,
            observation,
            frame_index,
            timestamp_seconds,
        )
        if packet.observation is None and self.anchor is None:
            return ready + [packet]

        dt = self._sample_dt(timestamp_seconds)
        if detected:
            self.last_detection_timestamp_seconds = timestamp_seconds
        self.delayed.append(packet)
        ready.extend(self._advance_filter(dt))
        if not detected and dropout_elapsed >= self.cfg.dropout_reset_seconds:
            ready.extend(self._end_segment())
        return ready

    def finish(self) -> list[_PendingFrame]:
        """Resolve the active segment with held-value padding."""
        return self._flush_segment()

    def _close_stale_segment_before_reacquisition(
        self,
        detected: bool,
        timestamp_seconds: float | None,
    ) -> list[_PendingFrame]:
        if not detected or self.anchor is None or not self.misses:
            return []
        # A low-rate or stalled source can jump from a missing sample straight
        # to reacquisition after the reset deadline. Close the old segment
        # before the new detection can enter its filter state.
        elapsed = self._dropout_elapsed(
            timestamp_seconds,
            include_current_step=timestamp_seconds is None,
        )
        return self._end_segment() if elapsed >= self.cfg.dropout_reset_seconds else []

    def _packet_for_sample(
        self,
        frame: np.ndarray,
        observation: FaceObservation | None,
        frame_index: int,
        timestamp_seconds: float | None,
    ) -> tuple[_PendingFrame, float]:
        if observation is not None:
            self.anchor = _snapshot_observation(observation)
            self.misses = 0
            packet = _PendingFrame(
                frame,
                self.anchor.to_observation(),
                raw_pose=self.anchor.sample.attitude,
                frame_index=frame_index,
                detected=True,
                screen_opacity=1.0,
            )
            return packet, 0.0

        if self.anchor is None:
            return _PendingFrame(frame, None, frame_index=frame_index), 0.0

        self.misses += 1
        elapsed = self._dropout_elapsed(timestamp_seconds)
        opacity = self._dropout_opacity(elapsed)
        held = self.anchor.to_observation() if opacity > 0 else None
        return (
            _PendingFrame(
                frame,
                held,
                frame_index=frame_index,
                detected=False,
                screen_opacity=opacity,
            ),
            elapsed,
        )

    def _compensation_latency(self) -> int:
        if not self.cfg.compensate_delay or self.cfg.webcam is not None or self.cfg.display:
            return 0
        return max(0, int(getattr(self.smoother, "group_delay", 0)))

    def _advance_filter(self, dt: float) -> list[_PendingFrame]:
        sample = self._filter_step(self._held_sample(), dt)
        if len(self.delayed) <= self.latency:
            return []
        return [self._resolve_packet_in_place(self.delayed.popleft(), sample)]

    def _filter_step(self, sample: _PoseSample, dt: float | None = None) -> _PoseSample:
        if getattr(self.smoother, "uses_timestamps", False):
            attitude = self.smoother.update(*sample.attitude, dt=dt)
            translation = (
                self.smoother.update_position(*sample.translation, dt=dt)
                if self.cfg.filter.smooth_translation
                else sample.translation
            )
        else:
            attitude = self.smoother.update(*sample.attitude)
            translation = (
                self.smoother.update_position(*sample.translation)
                if self.cfg.filter.smooth_translation
                else sample.translation
            )
        return _PoseSample(*attitude, *translation)

    def _resolve_packet_in_place(
        self,
        packet: _PendingFrame,
        sample: _PoseSample,
    ) -> _PendingFrame:
        self.diagnostics.record(packet.frame_index, packet.raw_pose, sample.attitude)
        observation = packet.observation
        if observation is None:
            return packet

        observation.pose.yaw = sample.yaw
        observation.pose.pitch = sample.pitch
        observation.pose.roll = sample.roll
        if self.cfg.filter.smooth_translation:
            observation.center = (sample.center_x, sample.center_y)
            observation.size = sample.face_size
        return packet

    def _flush_segment(self) -> list[_PendingFrame]:
        """Resolve delayed frames with held-value padding before reset or EOF."""
        ready = []
        while self.delayed:
            sample = self._filter_step(self._held_sample(), 1.0 / self.fps)
            ready.append(self._resolve_packet_in_place(self.delayed.popleft(), sample))
        return ready

    def _end_segment(self) -> list[_PendingFrame]:
        ready = self._flush_segment()
        self.diagnostics.end_segment()
        self.smoother.reset()
        self.anchor = None
        self.last_timestamp_seconds = None
        self.last_detection_timestamp_seconds = None
        self.misses = 0
        return ready

    @staticmethod
    def _validated_timestamp(timestamp_seconds: float | None) -> float | None:
        if timestamp_seconds is None:
            return None
        if (
            not isinstance(timestamp_seconds, (int, float))
            or isinstance(timestamp_seconds, bool)
            or not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0
        ):
            raise ValueError("frame timestamp must be a non-negative finite number")
        return float(timestamp_seconds)

    def _sample_dt(self, timestamp_seconds: float | None) -> float:
        nominal = 1.0 / self.fps
        if timestamp_seconds is None:
            return nominal
        if self.last_timestamp_seconds is None:
            dt = nominal
        else:
            dt = timestamp_seconds - self.last_timestamp_seconds
            if dt <= 0:
                raise ValueError("frame timestamps must be strictly increasing")
        self.last_timestamp_seconds = timestamp_seconds
        return dt

    def _check_timestamp_order(self, timestamp_seconds: float | None) -> None:
        if (
            timestamp_seconds is not None
            and self.last_timestamp_seconds is not None
            and timestamp_seconds <= self.last_timestamp_seconds
        ):
            raise ValueError("frame timestamps must be strictly increasing")

    def _dropout_elapsed(
        self,
        timestamp_seconds: float | None,
        *,
        include_current_step: bool = False,
    ) -> float:
        if timestamp_seconds is not None and self.last_detection_timestamp_seconds is not None:
            return timestamp_seconds - self.last_detection_timestamp_seconds
        steps = self.misses + int(include_current_step)
        return steps / self.fps

    def _dropout_opacity(self, elapsed: float) -> float:
        hold = self.cfg.dropout_hold_seconds
        reset = self.cfg.dropout_reset_seconds
        if elapsed >= reset:
            return 0.0
        if elapsed <= hold:
            return 1.0
        if reset == hold:
            return 0.0
        return (reset - elapsed) / (reset - hold)

    def _held_sample(self) -> _PoseSample:
        if self.anchor is None:
            raise RuntimeError("internal error: delayed samples have no source pose")
        return self.anchor.sample


def _snapshot_observation(observation: FaceObservation) -> _ObservationSnapshot:
    """Validate backend output before it can enter filter or renderer state."""
    try:
        sample = _PoseSample(
            float(observation.pose.yaw),
            float(observation.pose.pitch),
            float(observation.pose.roll),
            float(observation.center[0]),
            float(observation.center[1]),
            float(observation.size),
        )
        bbox = tuple(observation.bbox)
        numeric_bbox = tuple(float(value) for value in bbox)
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("pose backend returned a malformed observation") from error

    values: Sequence[float] = (*sample.attitude, *sample.translation)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose backend returned non-finite pose or face geometry")
    if sample.face_size <= 0:
        raise ValueError("pose backend returned a non-positive face size")
    if len(numeric_bbox) != 4 or not all(math.isfinite(value) for value in numeric_bbox):
        raise ValueError("pose backend returned an invalid bounding box")
    if numeric_bbox[0] > numeric_bbox[2] or numeric_bbox[1] > numeric_bbox[3]:
        raise ValueError("pose backend returned an inverted bounding box")

    normalized_bbox: BoundingBox = (bbox[0], bbox[1], bbox[2], bbox[3])
    return _ObservationSnapshot(sample, normalized_bbox, observation.landmarks)
