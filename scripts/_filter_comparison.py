"""Frozen temporal policies shared by the public filter benchmark and visual.

EMA is intentionally benchmark-only: it is a useful first-order reference, not
part of the maintained runtime or CLI.  The other policies adapt the exact
project configurations used by ``frame2frame``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from frame2frame.config import FilterConfig
from frame2frame.filters import create_filter


@dataclass(frozen=True)
class ComparisonPolicy:
    key: str
    label: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class ComparisonProfile:
    key: str
    label: str


@dataclass(frozen=True)
class ComparisonWorkload:
    key: str
    motion: ComparisonProfile
    noise: ComparisonProfile
    clean: np.ndarray
    noisy: np.ndarray


POLICIES = (
    ComparisonPolicy("none", "RAW OBSERVATION", {}),
    ComparisonPolicy("ema", "EMA 2.5 HZ", {"cutoff_hz": 2.5}),
    ComparisonPolicy(
        "fir",
        "FIR / ANGULAR DEFAULT",
        {
            "cutoff_hz": 2.5,
            "transition_hz": 5.0,
            "ripple_db": 60.0,
            "pitch_cutoff_scale": 0.5,
            "roll_cutoff_scale": 0.1,
        },
    ),
    ComparisonPolicy(
        "oneeuro",
        "ONE EURO / ANGULAR DEFAULT",
        {"min_cutoff": 1.0, "beta": 0.3, "d_cutoff": 1.0},
    ),
    ComparisonPolicy(
        "kalman",
        "KALMAN A100",
        {"acceleration_std": 100.0, "measurement_std": 1.0},
    ),
)

MOTION_PROFILES = (
    ComparisonProfile("smooth", "smooth multi-axis sweep"),
    ComparisonProfile("reversal", "stop-and-reverse motion"),
)

NOISE_PROFILES = (
    ComparisonProfile("white", "independent frame noise"),
    ComparisonProfile("correlated", "temporally correlated noise"),
    ComparisonProfile("burst", "sparse decaying outliers"),
    ComparisonProfile("sample-hold", "piecewise-constant tracking error"),
)

_NOISE_STD_DEG = np.array((1.25, 1.0, 1.6), dtype=np.float64)


def _motion_trace(key: str, frames: int, fps: float) -> np.ndarray:
    """Return one deterministic clean motion profile."""
    t = np.arange(frames, dtype=np.float64) / fps
    if key == "smooth":
        return np.column_stack(
            (
                18.0 * np.sin(2.0 * np.pi * 0.18 * t) + 4.0 * np.sin(2.0 * np.pi * 0.47 * t + 0.4),
                10.0 * np.sin(2.0 * np.pi * 0.21 * t + 0.8),
                7.0 * np.sin(2.0 * np.pi * 0.13 * t - 0.3),
            )
        )
    if key == "reversal":
        positions = np.array(
            (0.00, 0.10, 0.22, 0.31, 0.44, 0.51, 0.60, 0.68, 0.78, 0.88, 1.00),
            dtype=np.float64,
        )
        keyframes = np.array(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (24.0, -8.0, 7.0),
                (24.0, -8.0, 7.0),
                (-22.0, 12.0, -9.0),
                (-22.0, 12.0, -9.0),
                (28.0, -10.0, 10.0),
                (-18.0, 14.0, -8.0),
                (-18.0, 14.0, -8.0),
                (20.0, -6.0, 6.0),
                (0.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
        samples = np.linspace(0.0, 1.0, frames)
        segment = np.clip(np.searchsorted(positions, samples, side="right") - 1, 0, 9)
        local = (samples - positions[segment]) / (positions[segment + 1] - positions[segment])
        blend = local**3 * (10.0 + local * (-15.0 + 6.0 * local))
        return keyframes[segment] + blend[:, None] * (keyframes[segment + 1] - keyframes[segment])
    raise ValueError(f"unknown motion profile: {key!r}")


def _normalise_noise(values: np.ndarray) -> np.ndarray:
    """Give every noise family the same zero-mean per-axis RMS budget."""
    centered = values - np.mean(values, axis=0)
    scale = np.std(centered, axis=0)
    if np.any(scale <= 0.0):
        raise RuntimeError("noise profile produced a constant axis")
    return centered * (_NOISE_STD_DEG / scale)


def _noise_trace(key: str, frames: int, seed: int) -> np.ndarray:
    """Return one seeded, non-periodic observation-error profile."""
    rng = np.random.default_rng(seed)
    if key == "white":
        values = rng.normal(size=(frames, 3))
    elif key == "correlated":
        rho = 0.92
        innovations = rng.normal(size=(frames, 3))
        values = np.empty_like(innovations)
        values[0] = innovations[0]
        innovation_scale = math.sqrt(1.0 - rho * rho)
        for index in range(1, frames):
            values[index] = rho * values[index - 1] + innovation_scale * innovations[index]
    elif key == "burst":
        values = 0.10 * rng.normal(size=(frames, 3))
        candidates = np.arange(4, max(5, frames - 5))
        events = min(len(candidates), max(6, round(frames / 30)))
        starts = np.sort(rng.choice(candidates, size=events, replace=False))
        kernel = np.array((1.0, 0.70, 0.40, 0.18), dtype=np.float64)
        for start in starts:
            stop = min(frames, start + len(kernel))
            signs = rng.choice((-1.0, 1.0), size=3)
            amplitudes = signs * rng.uniform(0.80, 1.20, size=3)
            values[start:stop] += kernel[: stop - start, None] * amplitudes
    elif key == "sample-hold":
        values = np.empty((frames, 3), dtype=np.float64)
        cursor = 0
        while cursor < frames:
            stop = min(frames, cursor + int(rng.integers(2, 7)))
            values[cursor:stop] = rng.normal(size=3)
            cursor = stop
        values += 0.08 * rng.normal(size=values.shape)
    else:
        raise ValueError(f"unknown noise profile: {key!r}")
    return _normalise_noise(values)


def benchmark_workloads(
    frames_per_workload: int,
    fps: float,
    seed: int,
) -> list[ComparisonWorkload]:
    """Return the frozen two-motion by four-noise benchmark matrix."""
    motions = {
        profile.key: _motion_trace(profile.key, frames_per_workload, fps)
        for profile in MOTION_PROFILES
    }
    noises = {
        profile.key: _noise_trace(
            profile.key,
            frames_per_workload,
            int(np.random.SeedSequence((seed, index)).generate_state(1)[0]),
        )
        for index, profile in enumerate(NOISE_PROFILES)
    }
    return [
        ComparisonWorkload(
            key=f"{motion.key}-{noise.key}",
            motion=motion,
            noise=noise,
            clean=motions[motion.key].copy(),
            noisy=motions[motion.key] + noises[noise.key],
        )
        for motion in MOTION_PROFILES
        for noise in NOISE_PROFILES
    ]


def workload_receipt() -> list[dict[str, str]]:
    """Return the ordered public workload registry without generated samples."""
    return [
        {
            "key": f"{motion.key}-{noise.key}",
            "motion": motion.key,
            "motion_label": motion.label,
            "noise": noise.key,
            "noise_label": noise.label,
        }
        for motion in MOTION_PROFILES
        for noise in NOISE_PROFILES
    ]


def showcase_trace(
    frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a seamless clean/noisy attitude loop for the README visual."""
    phase = 2.0 * np.pi * np.arange(frames, dtype=np.float64) / frames
    clean = np.column_stack(
        (
            18.0 * np.sin(phase - 0.4) + 5.0 * np.sin(2.0 * phase + 0.7),
            9.0 * np.sin(phase + 0.8) + 2.0 * np.sin(3.0 * phase - 0.2),
            6.0 * np.sin(phase - 0.3) + 1.5 * np.sin(2.0 * phase + 1.2),
        )
    )

    rng = np.random.default_rng(seed)
    harmonics = np.array((21.0, 27.0, 33.0, 39.0), dtype=np.float64)
    noise = np.empty_like(clean)
    target_std = np.array((1.25, 1.0, 1.6), dtype=np.float64)
    for axis, scale in enumerate(target_std):
        weights = rng.uniform(0.65, 1.0, size=len(harmonics))
        offsets = rng.uniform(0.0, 2.0 * np.pi, size=len(harmonics))
        signal = np.sum(
            weights[:, None] * np.sin(harmonics[:, None] * phase[None, :] + offsets[:, None]),
            axis=0,
        )
        noise[:, axis] = signal * (scale / np.std(signal))
    return clean, clean + noise


def policy_receipt() -> list[dict[str, object]]:
    """Return a JSON-safe description of the frozen comparison policies."""
    return [
        {"key": policy.key, "label": policy.label, "parameters": dict(policy.parameters)}
        for policy in POLICIES
    ]


class EMAFilter:
    """Benchmark-only first-order causal low-pass parameterized in hertz."""

    uses_timestamps = True

    def __init__(self, fps: float, cutoff_hz: float = 2.5) -> None:
        self.fps = float(fps)
        self.cutoff_hz = float(cutoff_hz)
        self.reset()

    @property
    def group_delay(self) -> int:
        return 0

    def reset(self) -> None:
        self._values: dict[str, float | None] = dict.fromkeys(
            ("yaw", "pitch", "roll", "cx", "cy", "size")
        )
        self._raw_angles: dict[str, float | None] = dict.fromkeys(("yaw", "pitch", "roll"))

    def _interval(self, dt: float | None) -> float:
        interval = 1.0 / self.fps if dt is None else float(dt)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("EMA dt must be a positive finite number")
        return interval

    def _step(self, channel: str, raw: float, dt: float | None) -> float:
        interval = self._interval(dt)
        value = float(raw)
        if channel in self._raw_angles:
            previous_raw = self._raw_angles[channel]
            if previous_raw is not None:
                value = previous_raw + (value - previous_raw + 180.0) % 360.0 - 180.0
            self._raw_angles[channel] = value

        previous = self._values[channel]
        if previous is None:
            self._values[channel] = value
            return value

        tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        alpha = interval / (tau + interval)
        filtered = previous + alpha * (value - previous)
        self._values[channel] = filtered
        return filtered

    def update(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        *,
        dt: float | None = None,
    ) -> tuple[float, float, float]:
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
    ) -> tuple[float, float, float]:
        return (
            self._step("cx", cx, dt),
            self._step("cy", cy, dt),
            self._step("size", size, dt),
        )


def create_comparison_filter(key: str, fps: float):
    """Build one fresh policy instance from the frozen public registry."""
    policies = {policy.key: policy for policy in POLICIES}
    if key not in policies:
        raise ValueError(f"unknown comparison policy: {key!r}")
    policy = policies[key]
    if key == "ema":
        return EMAFilter(fps, cutoff_hz=policy.parameters["cutoff_hz"])
    return create_filter(
        fps,
        FilterConfig(
            kind=key,
            smooth_translation=False,
            **policy.parameters,
        ),
    )
