"""Temporal smoothing of the pose stream (yaw/pitch/roll plus face centre/size).

``fir`` is the project's default and namesake: a linear-phase Kaiser-window
low-pass run per axis (roll is smoothed hardest because it is the noisiest axis
out of the pose estimators). ``oneeuro`` trades a little jitter for much lower
latency and suits the live webcam path where group delay is felt.
"""

from __future__ import annotations

import math
from typing import Protocol, cast

import numpy as np
from scipy.signal import firwin, kaiserord, lfilter, lfilter_zi

from .config import FilterConfig

_ANGLE_CHANNELS = ("yaw", "pitch", "roll")
_POSITION_CHANNELS = ("cx", "cy", "size")  # face centre and size, in pixels
_CHANNELS = _ANGLE_CHANNELS + _POSITION_CHANNELS
PoseValues = tuple[float, float, float]


class _PoseFilter(Protocol):
    """Internal surface the temporal tracker needs from a pose filter."""

    @property
    def group_delay(self) -> int:
        """Return the constant offline delay in samples, or zero when absent."""
        ...

    def reset(self) -> None:
        """Discard all temporal state."""
        ...

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        """Filter one angular pose sample."""
        ...

    def update_position(
        self,
        cx: float,
        cy: float,
        size: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        """Filter one face-position sample."""
        ...


def _wrap_step(angle: float, last: float | None) -> float:
    """Keep an angle stream continuous across the +/-180 seam."""
    if last is None:
        return angle
    diff = angle - last
    if diff > 180:
        return angle - 360
    if diff < -180:
        return angle + 360
    return angle


class FIRAttitudeFilter:
    def __init__(self, fps: float, cfg: FilterConfig | None = None) -> None:
        cfg = cfg or FilterConfig()
        cfg.validate(fps)
        fps = float(fps)
        nyquist = fps / 2.0

        # kaiserord requires a normalized width no greater than one. Low-frame-rate
        # sources can make the configured transition wider than the entire Nyquist
        # band; capping the design width preserves a valid, short anti-jitter filter.
        normalized_width = min(cfg.transition_hz / nyquist, 0.999)
        n, beta = kaiserord(cfg.ripple_db, normalized_width)
        if n % 2 == 0:
            n += 1
        self.n: int = n
        cutoff_scales: dict[str, float] = {
            "yaw": 1.0,
            "pitch": cfg.pitch_cutoff_scale,
            "roll": cfg.roll_cutoff_scale,
        }
        cutoff_scales.update(dict.fromkeys(_POSITION_CHANNELS, 1.0))
        self._taps: dict[str, np.ndarray] = {
            channel: firwin(
                n,
                cfg.cutoff_hz * scale / nyquist,
                window=("kaiser", beta),
            )
            for channel, scale in cutoff_scales.items()
        }
        self.reset()

    @property
    def group_delay(self) -> int:
        """Constant lag the linear-phase filter introduces, in samples."""
        return (self.n - 1) // 2

    def reset(self) -> None:
        self._zi: dict[str, np.ndarray | None] = dict.fromkeys(_CHANNELS)
        self._last: dict[str, float | None] = dict.fromkeys(_CHANNELS)

    def _step(self, channel: str, value: float) -> float:
        if channel in _ANGLE_CHANNELS:  # pixel channels have no +/-180 seam
            value = _wrap_step(value, self._last[channel])
        self._last[channel] = value
        taps = self._taps[channel]
        state = self._zi[channel]
        if state is None:
            # Prime the state to steady state so the output doesn't ramp up
            # from zero over the first ~N samples.
            state = lfilter_zi(taps, 1.0) * value
        out, state = lfilter(taps, 1.0, [value], zi=state)
        self._zi[channel] = state
        return float(out[0])

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        del dt
        return (self._step("yaw", yaw), self._step("pitch", pitch), self._step("roll", roll))

    def update_position(
        self,
        cx: float,
        cy: float,
        size: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        del dt
        return (
            self._step("cx", cx),
            self._step("cy", cy),
            self._step("size", size),
        )


class OneEuroFilter:
    """Casiez et al. (2012). Speed-adaptive low-pass: more smoothing when still,
    less lag when moving fast."""

    # Retained as an introspection hint for existing experiments. The tracker
    # no longer needs to branch on it because every built-in filter accepts dt.
    uses_timestamps = True

    def __init__(self, fps: float, cfg: FilterConfig | None = None) -> None:
        cfg = cfg or FilterConfig()
        cfg.validate(fps)
        self.fps: float = float(fps)
        self.min_cutoff: float = cfg.min_cutoff
        self.beta: float = cfg.beta
        self.d_cutoff: float = cfg.d_cutoff
        self.reset()

    @property
    def group_delay(self) -> int:
        return 0

    def reset(self) -> None:
        self._x: dict[str, float | None] = dict.fromkeys(_CHANNELS)
        self._dx: dict[str, float] = dict.fromkeys(_CHANNELS, 0.0)
        self._last: dict[str, float | None] = dict.fromkeys(_CHANNELS)

    def _validated_dt(self, dt: float | None) -> float:
        interval = 1.0 / self.fps if dt is None else float(dt)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("One Euro dt must be a positive finite number")
        return interval

    @staticmethod
    def _alpha_for_dt(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _alpha(self, cutoff: float, dt: float | None = None) -> float:
        return self._alpha_for_dt(cutoff, self._validated_dt(dt))

    def _step(self, channel: str, value: float, dt: float | None = None) -> float:
        dt = self._validated_dt(dt)
        previous = self._last[channel]
        if channel in _ANGLE_CHANNELS:
            value = _wrap_step(value, previous)
        self._last[channel] = value
        filtered = self._x[channel]
        if filtered is None:
            self._x[channel] = value
            return value
        # The One Euro paper filters the derivative of consecutive raw
        # samples, not the distance from the previously smoothed output.
        previous = cast(float, previous)  # narrowed by the first-sample return above
        dx = (value - previous) / dt
        self._dx[channel] += self._alpha_for_dt(self.d_cutoff, dt) * (dx - self._dx[channel])
        cutoff = self.min_cutoff + self.beta * abs(self._dx[channel])
        filtered += self._alpha_for_dt(cutoff, dt) * (value - filtered)
        self._x[channel] = filtered
        return filtered

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        return (
            self._step("yaw", yaw, dt),
            self._step("pitch", pitch, dt),
            self._step("roll", roll, dt),
        )

    def update_position(
        self,
        cx: float,
        cy: float,
        size: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        return (
            self._step("cx", cx, dt),
            self._step("cy", cy, dt),
            self._step("size", size, dt),
        )


class PassThrough:
    @property
    def group_delay(self) -> int:
        return 0

    def reset(self) -> None:
        pass

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        del dt
        return yaw, pitch, roll

    def update_position(
        self,
        cx: float,
        cy: float,
        size: float,
        *,
        dt: float | None = None,
    ) -> PoseValues:
        del dt
        return cx, cy, size


def create_filter(fps: float, cfg: FilterConfig | None = None) -> _PoseFilter:
    cfg = cfg or FilterConfig()
    kind = cfg.kind.strip().lower() if isinstance(cfg.kind, str) else ""
    if kind == "fir":
        return FIRAttitudeFilter(fps, cfg)
    if kind == "oneeuro":
        return OneEuroFilter(fps, cfg)
    cfg.validate(fps)
    return PassThrough()
