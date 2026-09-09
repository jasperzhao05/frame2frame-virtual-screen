"""Deterministic, model-free benchmark for five temporal policies.

The benchmark separates observation noise from intentional head motion with a
balanced two-motion by four-noise stress matrix.  Every noisy cell is paired
with its clean control.  Noise metrics are macro-averaged across the eight
cells; Screen Age and motion distortion are evaluated once per clean motion.
Angle-space step response and throughput remain in the machine-readable
receipt.

No camera, video, model weights, or network access is required.

Examples:

    python -m scripts.benchmark_smoothing
    python -m scripts.benchmark_smoothing --format markdown
    python -m scripts.benchmark_smoothing --format json --output output/benchmark.json
    python -m scripts.benchmark_smoothing --check
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from numbers import Real
from pathlib import Path
from typing import TypedDict

import numpy as np

from frame2frame import render
from frame2frame.config import ScreenConfig
from frame2frame.pose.base import FaceObservation, HeadPose
from scripts._filter_comparison import (
    POLICIES,
    ComparisonWorkload,
    benchmark_workloads,
    create_comparison_filter,
    policy_receipt,
    workload_receipt,
)

_FILTERS = tuple(policy.key for policy in POLICIES)
_AXES = ("yaw", "pitch", "roll")
_SCREEN_SIZE = (480, 320)
_SCREEN_CENTER = (240.0, 160.0)
_SCREEN_FACE_SIZE = 40.0
_SCREEN_DISTANCE = 4.0
_SCREEN_AGE_WINDOW_SECONDS = 2.0 / 3.0
_DEFAULT_FRAMES = 1800
_DEFAULT_FPS = 30.0
_DEFAULT_SEED = 20260730
_WORKLOAD_RECEIPT = workload_receipt()
_WORKLOAD_COUNT = len(_WORKLOAD_RECEIPT)


class WorkloadRow(TypedDict):
    workload: str
    motion: str
    noise: str
    input_jitter_rms_deg: float
    input_screen_jitter_rms_px: float
    input_screen_jitter_p95_px: float
    jitter_rms_deg: float
    jitter_ratio: float
    jitter_reduction_pct: float
    axis_jitter_rms_deg: dict[str, float]
    screen_jitter_rms_px: float
    screen_jitter_ratio: float
    screen_jitter_reduction_pct: float
    screen_jitter_p95_px: float
    screen_jitter_p95_ratio: float
    screen_jitter_p95_reduction_pct: float
    screen_current_rmse_px: float
    screen_current_p95_px: float


class MotionRow(TypedDict):
    motion: str
    motion_rmse_deg_aligned: float
    screen_age_frames: int
    screen_age_ms: float
    screen_age_censored: bool
    screen_motion_rmse_px_aligned: float


class BenchmarkRow(TypedDict):
    filter: str
    macro_jitter_rms_deg: float
    macro_jitter_ratio: float
    macro_jitter_reduction_pct: float
    macro_axis_jitter_rms_deg: dict[str, float]
    latency_frames_50pct: int
    latency_ms_50pct: float
    designed_group_delay_frames: int
    macro_motion_rmse_deg_aligned: float
    macro_screen_jitter_rms_px: float
    macro_screen_jitter_ratio: float
    macro_screen_jitter_reduction_pct: float
    macro_screen_jitter_p95_ratio: float
    macro_screen_jitter_p95_reduction_pct: float
    macro_screen_age_frames: float
    screen_age_range_frames: list[int]
    macro_screen_motion_rmse_px_aligned: float
    worst_screen_jitter_reduction_pct: float
    worst_screen_motion_rmse_px_aligned: float
    macro_screen_current_rmse_px: float
    worst_screen_current_p95_px: float
    throughput_attitudes_s: float
    workloads: list[WorkloadRow]
    motions: list[MotionRow]


class BenchmarkMethod(TypedDict):
    total_frames: int
    frames_per_workload: int
    workload_count: int
    fps: float
    seed: int
    speed_samples: int
    evaluation_warmup_frames_per_workload: int
    macro_input_jitter_rms_deg: float
    macro_input_screen_jitter_rms_px: float
    macro_input_screen_jitter_p95_px: float
    macro_definition: str
    latency_definition: str
    jitter_definition: str
    screen_metric_definition: str
    tail_metric_definition: str
    current_error_definition: str
    noise_axis_std_deg: list[float]
    screen_carrier: dict[str, object]
    policies: list[dict[str, object]]
    workloads: list[dict[str, str]]


class BenchmarkEnvironment(TypedDict):
    python: str
    platform: str
    numpy: str


class BenchmarkReport(TypedDict):
    schema_version: int
    method: BenchmarkMethod
    environment: BenchmarkEnvironment
    results: list[BenchmarkRow]


def _project_screens(values: np.ndarray) -> np.ndarray:
    width, height = _SCREEN_SIZE
    frame = np.empty((height, width, 3), dtype=np.uint8)
    config = ScreenConfig(distance_mul=_SCREEN_DISTANCE)
    quads = np.empty((len(values), 4, 2), dtype=np.float64)
    for index, angles in enumerate(values):
        observation = FaceObservation(
            pose=HeadPose(*angles),
            center=_SCREEN_CENTER,
            size=_SCREEN_FACE_SIZE,
            bbox=(0.0, 0.0, 0.0, 0.0),
        )
        projection = render._project_screen(frame, observation, config)
        if projection is None:
            raise RuntimeError("frozen screen workload produced an invalid projection")
        quads[index] = projection.quad
    return quads


def _screen_age_and_error(
    filtered: np.ndarray,
    reference: np.ndarray,
    *,
    trim: int,
    max_age: int,
) -> tuple[int, float, bool]:
    errors = []
    for age in range(max_age + 1):
        start = max(trim, age)
        stop = len(reference) - age if age else len(reference)
        residual = filtered[start:] - reference[start - age : stop]
        errors.append(float(np.mean(np.sum(np.square(residual), axis=-1))))
    age = int(np.argmin(errors))
    return age, math.sqrt(errors[age]), age == max_age


def _screen_age_search_limit(fps: float, delays: dict[str, int]) -> int:
    return max(round(fps * _SCREEN_AGE_WINDOW_SECONDS), max(delays.values()) + 1)


def _corner_rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(residual), axis=-1))))


def _corner_p95(residual: np.ndarray) -> float:
    per_frame = np.sqrt(np.mean(np.sum(np.square(residual), axis=-1), axis=-1))
    return float(np.percentile(per_frame, 95))


def _apply(kind: str, values: np.ndarray, fps: float) -> tuple[np.ndarray, int]:
    filt = create_comparison_filter(kind, fps)
    output = np.empty_like(values)
    for index, sample in enumerate(values):
        output[index] = filt.update(*sample)
    return output, int(getattr(filt, "group_delay", 0))


def _step_latency(kind: str, fps: float) -> tuple[int, int]:
    """Measure first crossing of 50% of a 20-degree yaw step."""
    step_at = round(2.0 * fps)
    frames = round(6.0 * fps)
    signal = np.zeros((frames, 3), dtype=np.float64)
    signal[step_at:, 0] = 20.0
    output, designed_delay = _apply(kind, signal, fps)
    crossing = np.flatnonzero(output[step_at:, 0] >= 10.0)
    latency_frames = int(crossing[0]) if crossing.size else frames - step_at
    return latency_frames, designed_delay


def _aligned_rmse(
    output: np.ndarray,
    clean: np.ndarray,
    latency: int,
    trim: int,
) -> float:
    start = max(trim, latency)
    if latency:
        lhs = output[start:]
        rhs = clean[start - latency : len(clean) - latency]
    else:
        lhs = output[start:]
        rhs = clean[start:]
    return float(np.sqrt(np.mean(np.square(lhs - rhs))))


def _attitude_throughput(kind: str, fps: float, samples: int) -> float:
    filt = create_comparison_filter(kind, fps)
    t0 = time.perf_counter()
    checksum = 0.0
    for index in range(samples):
        phase = index * 0.017
        yaw, pitch, roll = filt.update(
            18.0 * math.sin(phase),
            9.0 * math.sin(phase * 0.7 + 0.2),
            5.0 * math.sin(phase * 0.4 - 0.3),
        )
        checksum += yaw + pitch + roll
    elapsed = time.perf_counter() - t0
    # Keep the loop observable without printing a meaningless implementation detail.
    if not math.isfinite(checksum):
        raise RuntimeError("non-finite filter output")
    return samples / elapsed


def _evaluate_workload(
    kind: str,
    workload: ComparisonWorkload,
    *,
    fps: float,
    trim: int,
    clean_out: np.ndarray,
    clean_output_screens: np.ndarray,
    reference_screens: np.ndarray,
    input_jitter: float,
    input_screen_jitter: float,
    input_screen_jitter_p95: float,
) -> WorkloadRow:
    noisy_out, _ = _apply(kind, workload.noisy, fps)
    residual = noisy_out[trim:] - clean_out[trim:]
    axis_jitter = np.sqrt(np.mean(np.square(residual), axis=0))
    jitter = float(np.sqrt(np.mean(np.square(residual))))
    noisy_output_screens = _project_screens(noisy_out)
    screen_residual = noisy_output_screens[trim:] - clean_output_screens[trim:]
    screen_jitter = _corner_rms(screen_residual)
    screen_jitter_p95 = _corner_p95(screen_residual)
    current_residual = noisy_output_screens[trim:] - reference_screens[trim:]
    return {
        "workload": workload.key,
        "motion": workload.motion.key,
        "noise": workload.noise.key,
        "input_jitter_rms_deg": input_jitter,
        "input_screen_jitter_rms_px": input_screen_jitter,
        "input_screen_jitter_p95_px": input_screen_jitter_p95,
        "jitter_rms_deg": jitter,
        "jitter_ratio": jitter / input_jitter,
        "jitter_reduction_pct": 100.0 * (1.0 - jitter / input_jitter),
        "axis_jitter_rms_deg": {axis: float(value) for axis, value in zip(_AXES, axis_jitter)},
        "screen_jitter_rms_px": screen_jitter,
        "screen_jitter_ratio": screen_jitter / input_screen_jitter,
        "screen_jitter_reduction_pct": 100.0 * (1.0 - screen_jitter / input_screen_jitter),
        "screen_jitter_p95_px": screen_jitter_p95,
        "screen_jitter_p95_ratio": screen_jitter_p95 / input_screen_jitter_p95,
        "screen_jitter_p95_reduction_pct": 100.0
        * (1.0 - screen_jitter_p95 / input_screen_jitter_p95),
        "screen_current_rmse_px": _corner_rms(current_residual),
        "screen_current_p95_px": _corner_p95(current_residual),
    }


def _mean(rows: list[WorkloadRow] | list[MotionRow], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def run_benchmark(
    *,
    frames: int = _DEFAULT_FRAMES,
    fps: float = _DEFAULT_FPS,
    seed: int = _DEFAULT_SEED,
    speed_samples: int = 10_000,
) -> BenchmarkReport:
    if frames % _WORKLOAD_COUNT:
        raise ValueError(f"frames must be divisible by the {_WORKLOAD_COUNT} workloads")
    frames_per_workload = frames // _WORKLOAD_COUNT
    minimum_frames = max(120, round(fps * 4.0))
    if frames_per_workload < minimum_frames:
        raise ValueError(
            f"each workload requires at least {minimum_frames} frames; received "
            f"{frames_per_workload}"
        )

    workloads = benchmark_workloads(frames_per_workload, fps, seed)
    delays = {
        kind: int(getattr(create_comparison_filter(kind, fps), "group_delay", 0))
        for kind in _FILTERS
    }
    latencies = {kind: _step_latency(kind, fps)[0] for kind in _FILTERS}
    trim = max(round(fps), 2 * max(delays.values()))
    max_screen_age = _screen_age_search_limit(fps, delays)

    clean_by_motion: dict[str, np.ndarray] = {}
    for workload in workloads:
        clean_by_motion.setdefault(workload.motion.key, workload.clean)
    clean_screens = {motion: _project_screens(clean) for motion, clean in clean_by_motion.items()}

    input_metrics: dict[str, tuple[float, float, float]] = {}
    for workload in workloads:
        noisy_screens = _project_screens(workload.noisy)
        input_jitter = float(
            np.sqrt(np.mean(np.square(workload.noisy[trim:] - workload.clean[trim:])))
        )
        screen_residual = noisy_screens[trim:] - clean_screens[workload.motion.key][trim:]
        input_metrics[workload.key] = (
            input_jitter,
            _corner_rms(screen_residual),
            _corner_p95(screen_residual),
        )

    results: list[BenchmarkRow] = []
    for kind in _FILTERS:
        clean_outputs: dict[str, np.ndarray] = {}
        clean_output_screens: dict[str, np.ndarray] = {}
        motion_rows: list[MotionRow] = []
        for motion, clean in clean_by_motion.items():
            clean_out, _ = _apply(kind, clean, fps)
            output_screens = _project_screens(clean_out)
            screen_age, screen_motion_rmse, screen_age_censored = _screen_age_and_error(
                output_screens,
                clean_screens[motion],
                trim=trim,
                max_age=max_screen_age,
            )
            clean_outputs[motion] = clean_out
            clean_output_screens[motion] = output_screens
            motion_rows.append(
                {
                    "motion": motion,
                    "motion_rmse_deg_aligned": _aligned_rmse(
                        clean_out,
                        clean,
                        latencies[kind],
                        trim,
                    ),
                    "screen_age_frames": screen_age,
                    "screen_age_ms": screen_age / fps * 1000.0,
                    "screen_age_censored": screen_age_censored,
                    "screen_motion_rmse_px_aligned": screen_motion_rmse,
                }
            )

        rows: list[WorkloadRow] = []
        for workload in workloads:
            input_jitter, input_screen_jitter, input_screen_jitter_p95 = input_metrics[workload.key]
            rows.append(
                _evaluate_workload(
                    kind,
                    workload,
                    fps=fps,
                    trim=trim,
                    clean_out=clean_outputs[workload.motion.key],
                    clean_output_screens=clean_output_screens[workload.motion.key],
                    reference_screens=clean_screens[workload.motion.key],
                    input_jitter=input_jitter,
                    input_screen_jitter=input_screen_jitter,
                    input_screen_jitter_p95=input_screen_jitter_p95,
                )
            )
        jitter_ratio = _mean(rows, "jitter_ratio")
        screen_jitter_ratio = _mean(rows, "screen_jitter_ratio")
        screen_jitter_p95_ratio = _mean(rows, "screen_jitter_p95_ratio")
        ages = [row["screen_age_frames"] for row in motion_rows]
        results.append(
            {
                "filter": kind,
                "macro_jitter_rms_deg": _mean(rows, "jitter_rms_deg"),
                "macro_jitter_ratio": jitter_ratio,
                "macro_jitter_reduction_pct": 100.0 * (1.0 - jitter_ratio),
                "macro_axis_jitter_rms_deg": {
                    axis: float(np.mean([row["axis_jitter_rms_deg"][axis] for row in rows]))
                    for axis in _AXES
                },
                "latency_frames_50pct": latencies[kind],
                "latency_ms_50pct": latencies[kind] / fps * 1000.0,
                "designed_group_delay_frames": delays[kind],
                "macro_motion_rmse_deg_aligned": _mean(motion_rows, "motion_rmse_deg_aligned"),
                "macro_screen_jitter_rms_px": _mean(rows, "screen_jitter_rms_px"),
                "macro_screen_jitter_ratio": screen_jitter_ratio,
                "macro_screen_jitter_reduction_pct": 100.0 * (1.0 - screen_jitter_ratio),
                "macro_screen_jitter_p95_ratio": screen_jitter_p95_ratio,
                "macro_screen_jitter_p95_reduction_pct": 100.0 * (1.0 - screen_jitter_p95_ratio),
                "macro_screen_age_frames": float(np.mean(ages)),
                "screen_age_range_frames": [min(ages), max(ages)],
                "macro_screen_motion_rmse_px_aligned": _mean(
                    motion_rows, "screen_motion_rmse_px_aligned"
                ),
                "worst_screen_jitter_reduction_pct": min(
                    row["screen_jitter_reduction_pct"] for row in rows
                ),
                "worst_screen_motion_rmse_px_aligned": max(
                    row["screen_motion_rmse_px_aligned"] for row in motion_rows
                ),
                "macro_screen_current_rmse_px": _mean(rows, "screen_current_rmse_px"),
                "worst_screen_current_p95_px": max(row["screen_current_p95_px"] for row in rows),
                "throughput_attitudes_s": _attitude_throughput(kind, fps, speed_samples),
                "workloads": rows,
                "motions": motion_rows,
            }
        )

    raw_rows = results[0]["workloads"]
    return {
        "schema_version": 5,
        "method": {
            "total_frames": frames,
            "frames_per_workload": frames_per_workload,
            "workload_count": _WORKLOAD_COUNT,
            "fps": fps,
            "seed": seed,
            "speed_samples": speed_samples,
            "evaluation_warmup_frames_per_workload": trim,
            "macro_input_jitter_rms_deg": _mean(raw_rows, "input_jitter_rms_deg"),
            "macro_input_screen_jitter_rms_px": _mean(raw_rows, "input_screen_jitter_rms_px"),
            "macro_input_screen_jitter_p95_px": _mean(raw_rows, "input_screen_jitter_p95_px"),
            "macro_definition": (
                "noise ratios: unweighted mean of eight motion-by-noise cells; "
                "clean-motion metrics: unweighted mean of two unique motion profiles"
            ),
            "latency_definition": "first output crossing of 50% of a 20-degree yaw step",
            "jitter_definition": (
                "RMS difference between noisy-input and clean-input filter outputs"
            ),
            "screen_metric_definition": (
                "RMS corresponding-corner distance after distance-4 projection"
            ),
            "tail_metric_definition": (
                "95th percentile per-frame RMS corresponding-corner distance"
            ),
            "current_error_definition": (
                "noisy-input filtered screen versus unfiltered clean screen at the same "
                "source frame; includes noise, motion distortion and lag; no age alignment"
            ),
            "screen_carrier": {
                "frame_size": list(_SCREEN_SIZE),
                "center_px": list(_SCREEN_CENTER),
                "face_size_px": _SCREEN_FACE_SIZE,
                "distance": _SCREEN_DISTANCE,
                "max_age_frames": max_screen_age,
                "max_age_seconds": max_screen_age / fps,
            },
            "noise_axis_std_deg": [1.25, 1.0, 1.6],
            "policies": policy_receipt(),
            "workloads": _WORKLOAD_RECEIPT,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "results": results,
    }


def _table(report: BenchmarkReport) -> str:
    lines = [
        "policy    macro RMS reject   worst cell   macro p95 reject   "
        "screen age       aligned error",
        "--------  -----------------  -----------  -----------------  "
        "---------------  -------------",
    ]
    for row in report["results"]:
        age_min, age_max = row["screen_age_range_frames"]
        age = f"{row['macro_screen_age_frames']:.1f} fr"
        if age_min != age_max:
            age += f" [{age_min}-{age_max}]"
        lines.append(
            f"{row['filter']:<8}  "
            f"{row['macro_screen_jitter_reduction_pct']:>15.1f}%  "
            f"{row['worst_screen_jitter_reduction_pct']:>9.1f}%  "
            f"{row['macro_screen_jitter_p95_reduction_pct']:>15.1f}%  "
            f"{age:>15}  "
            f"{row['macro_screen_motion_rmse_px_aligned']:>9.3f} px"
        )
    method = report["method"]
    lines.extend(
        (
            "",
            f"seed={method['seed']}  workloads={method['workload_count']}  "
            f"total frames={method['total_frames']}  "
            f"fps={method['fps']:.1f}  "
            f"macro input RMS={method['macro_input_screen_jitter_rms_px']:.3f} px  "
            f"macro input p95={method['macro_input_screen_jitter_p95_px']:.3f} px",
            "Attitude throughput is machine-dependent; quality inputs are deterministic.",
        )
    )
    return "\n".join(lines)


def _markdown(report: BenchmarkReport) -> str:
    lines = [
        "| Policy | Macro RMS reduction | Worst-cell reduction | Macro p95 reduction | "
        "Screen Age macro [range] | Macro aligned error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        age_min, age_max = row["screen_age_range_frames"]
        age = f"{row['macro_screen_age_frames']:.1f} frames"
        if age_min != age_max:
            age += f" [{age_min}–{age_max}]"
        lines.append(
            f"| `{row['filter']}` | {row['macro_screen_jitter_reduction_pct']:.1f}% | "
            f"{row['worst_screen_jitter_reduction_pct']:.1f}% | "
            f"{row['macro_screen_jitter_p95_reduction_pct']:.1f}% | {age} | "
            f"{row['macro_screen_motion_rmse_px_aligned']:.3f} px |"
        )
    return "\n".join(lines)


def _nonfinite_numeric_paths(value: object, path: str = "report") -> list[str]:
    """Return paths to non-finite numeric values in a benchmark receipt."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return [] if math.isfinite(float(value)) else [path]
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_nonfinite_numeric_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nonfinite_numeric_paths(item, f"{path}[{index}]"))
        return paths
    return []


def _input_baseline_failures(
    rows: dict[str, BenchmarkRow],
    raw: BenchmarkRow,
) -> list[str]:
    failures = []
    for row in rows.values():
        for baseline, item in zip(raw["workloads"], row["workloads"]):
            consistent = (
                math.isclose(
                    item["input_jitter_rms_deg"],
                    baseline["input_jitter_rms_deg"],
                    abs_tol=1e-12,
                )
                and math.isclose(
                    item["input_screen_jitter_rms_px"],
                    baseline["input_screen_jitter_rms_px"],
                    abs_tol=1e-12,
                )
                and math.isclose(
                    item["input_screen_jitter_p95_px"],
                    baseline["input_screen_jitter_p95_px"],
                    abs_tol=1e-12,
                )
            )
            if not consistent:
                failures.append(f"{row['filter']}/{item['workload']} input baseline changed")
    return failures


def _self_check(report: BenchmarkReport, *, reference_thresholds: bool = True) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(report["schema_version"] == 5, "unexpected benchmark schema version")
    require(
        report["method"]["workload_count"] == _WORKLOAD_COUNT,
        "unexpected workload count",
    )
    require(
        report["method"]["workloads"] == _WORKLOAD_RECEIPT,
        "workload registry changed",
    )
    require(
        report["method"]["total_frames"]
        == report["method"]["frames_per_workload"] * _WORKLOAD_COUNT,
        "total frame accounting is inconsistent",
    )

    rows = {row["filter"]: row for row in report["results"]}
    require(list(rows) == list(_FILTERS), "policy result order changed")
    expected_workloads = [item["key"] for item in _WORKLOAD_RECEIPT]
    expected_motions = list(dict.fromkeys(item["motion"] for item in _WORKLOAD_RECEIPT))

    for kind, row in rows.items():
        require(
            [item["workload"] for item in row["workloads"]] == expected_workloads,
            f"{kind} workload rows changed",
        )
        require(
            [item["motion"] for item in row["motions"]] == expected_motions,
            f"{kind} clean-motion rows changed",
        )

        noise_rows = row["workloads"]
        motion_rows = row["motions"]
        require(
            math.isclose(
                row["macro_screen_current_rmse_px"], _mean(noise_rows, "screen_current_rmse_px")
            ),
            f"{kind} current-frame error is not the workload mean",
        )
        require(
            math.isclose(
                row["worst_screen_current_p95_px"],
                max(item["screen_current_p95_px"] for item in noise_rows),
            ),
            f"{kind} current-frame tail is not the worst workload",
        )
        for item in noise_rows:
            angle_ratio = item["jitter_rms_deg"] / item["input_jitter_rms_deg"]
            screen_ratio = item["screen_jitter_rms_px"] / item["input_screen_jitter_rms_px"]
            p95_ratio = item["screen_jitter_p95_px"] / item["input_screen_jitter_p95_px"]
            require(
                math.isclose(item["jitter_ratio"], angle_ratio, abs_tol=1e-12)
                and math.isclose(
                    item["jitter_reduction_pct"],
                    100.0 * (1.0 - angle_ratio),
                    abs_tol=1e-10,
                ),
                f"{kind}/{item['workload']} angle ratio is inconsistent",
            )
            require(
                math.isclose(item["screen_jitter_ratio"], screen_ratio, abs_tol=1e-12)
                and math.isclose(
                    item["screen_jitter_reduction_pct"],
                    100.0 * (1.0 - screen_ratio),
                    abs_tol=1e-10,
                ),
                f"{kind}/{item['workload']} screen RMS ratio is inconsistent",
            )
            require(
                math.isclose(item["screen_jitter_p95_ratio"], p95_ratio, abs_tol=1e-12)
                and math.isclose(
                    item["screen_jitter_p95_reduction_pct"],
                    100.0 * (1.0 - p95_ratio),
                    abs_tol=1e-10,
                ),
                f"{kind}/{item['workload']} screen p95 ratio is inconsistent",
            )
        for item in motion_rows:
            require(
                math.isclose(
                    item["screen_age_ms"],
                    item["screen_age_frames"] / report["method"]["fps"] * 1000.0,
                    abs_tol=1e-12,
                ),
                f"{kind}/{item['motion']} Screen Age milliseconds are inconsistent",
            )
            require(
                item["screen_age_censored"]
                == (
                    item["screen_age_frames"]
                    == report["method"]["screen_carrier"]["max_age_frames"]
                ),
                f"{kind}/{item['motion']} Screen Age censor flag is inconsistent",
            )
            require(
                not item["screen_age_censored"],
                f"{kind}/{item['motion']} Screen Age hit the search boundary",
            )

        macro_angle_ratio = float(np.mean([item["jitter_ratio"] for item in noise_rows]))
        macro_ratio = float(np.mean([item["screen_jitter_ratio"] for item in noise_rows]))
        macro_p95_ratio = float(np.mean([item["screen_jitter_p95_ratio"] for item in noise_rows]))
        ages = [item["screen_age_frames"] for item in motion_rows]
        require(
            math.isclose(
                row["macro_jitter_rms_deg"],
                float(np.mean([item["jitter_rms_deg"] for item in noise_rows])),
                abs_tol=1e-12,
            )
            and math.isclose(row["macro_jitter_ratio"], macro_angle_ratio, abs_tol=1e-12)
            and math.isclose(
                row["macro_jitter_reduction_pct"],
                100.0 * (1.0 - macro_angle_ratio),
                abs_tol=1e-10,
            ),
            f"{kind} macro angle metrics are inconsistent",
        )
        for axis in _AXES:
            require(
                math.isclose(
                    row["macro_axis_jitter_rms_deg"][axis],
                    float(np.mean([item["axis_jitter_rms_deg"][axis] for item in noise_rows])),
                    abs_tol=1e-12,
                ),
                f"{kind} macro {axis} jitter is inconsistent",
            )
        require(
            math.isclose(
                row["macro_screen_jitter_rms_px"],
                float(np.mean([item["screen_jitter_rms_px"] for item in noise_rows])),
                abs_tol=1e-12,
            ),
            f"{kind} macro screen-jitter RMS is inconsistent",
        )
        require(
            math.isclose(row["macro_screen_jitter_ratio"], macro_ratio, abs_tol=1e-12),
            f"{kind} macro screen-jitter ratio is not the equal-cell mean",
        )
        require(
            math.isclose(
                row["macro_screen_jitter_reduction_pct"],
                100.0 * (1.0 - macro_ratio),
                abs_tol=1e-10,
            ),
            f"{kind} macro screen-jitter reduction is inconsistent",
        )
        require(
            math.isclose(
                row["macro_screen_jitter_p95_ratio"],
                macro_p95_ratio,
                abs_tol=1e-12,
            ),
            f"{kind} macro p95 ratio is not the equal-cell mean",
        )
        require(
            math.isclose(
                row["macro_screen_jitter_p95_reduction_pct"],
                100.0 * (1.0 - macro_p95_ratio),
                abs_tol=1e-10,
            ),
            f"{kind} macro p95 reduction is inconsistent",
        )
        require(
            math.isclose(
                row["worst_screen_jitter_reduction_pct"],
                min(item["screen_jitter_reduction_pct"] for item in noise_rows),
                abs_tol=1e-10,
            ),
            f"{kind} worst-cell reduction is inconsistent",
        )
        require(
            math.isclose(row["macro_screen_age_frames"], float(np.mean(ages))),
            f"{kind} Screen Age macro repeats or omits a clean motion",
        )
        require(
            row["screen_age_range_frames"] == [min(ages), max(ages)],
            f"{kind} Screen Age range is inconsistent",
        )
        require(
            math.isclose(
                row["macro_motion_rmse_deg_aligned"],
                float(np.mean([item["motion_rmse_deg_aligned"] for item in motion_rows])),
                abs_tol=1e-12,
            ),
            f"{kind} clean-motion angle error is not the two-motion mean",
        )
        require(
            math.isclose(
                row["macro_screen_motion_rmse_px_aligned"],
                float(np.mean([item["screen_motion_rmse_px_aligned"] for item in motion_rows])),
                abs_tol=1e-12,
            ),
            f"{kind} clean-motion aligned error is not the two-motion mean",
        )
        require(
            math.isclose(
                row["worst_screen_motion_rmse_px_aligned"],
                max(item["screen_motion_rmse_px_aligned"] for item in motion_rows),
                abs_tol=1e-12,
            ),
            f"{kind} worst clean-motion error is inconsistent",
        )
        require(
            math.isclose(
                row["latency_ms_50pct"],
                row["latency_frames_50pct"] / report["method"]["fps"] * 1000.0,
                abs_tol=1e-12,
            ),
            f"{kind} step-latency milliseconds are inconsistent",
        )

    raw = rows["none"]
    require(
        math.isclose(
            report["method"]["macro_input_jitter_rms_deg"],
            float(np.mean([item["input_jitter_rms_deg"] for item in raw["workloads"]])),
            abs_tol=1e-12,
        ),
        "method macro input-angle jitter is inconsistent",
    )
    require(
        math.isclose(
            report["method"]["macro_input_screen_jitter_rms_px"],
            float(np.mean([item["input_screen_jitter_rms_px"] for item in raw["workloads"]])),
            abs_tol=1e-12,
        ),
        "method macro input-screen RMS is inconsistent",
    )
    require(
        math.isclose(
            report["method"]["macro_input_screen_jitter_p95_px"],
            float(np.mean([item["input_screen_jitter_p95_px"] for item in raw["workloads"]])),
            abs_tol=1e-12,
        ),
        "method macro input-screen p95 is inconsistent",
    )
    failures.extend(_input_baseline_failures(rows, raw))
    for workload in raw["workloads"]:
        require(
            abs(workload["screen_jitter_ratio"] - 1.0) <= 1e-12
            and abs(workload["screen_jitter_reduction_pct"]) <= 1e-10
            and abs(workload["screen_jitter_p95_ratio"] - 1.0) <= 1e-12,
            f"pass-through changed workload {workload['workload']}",
        )
    for motion in raw["motions"]:
        require(
            motion["screen_age_frames"] == 0 and motion["screen_motion_rmse_px_aligned"] <= 1e-12,
            f"pass-through changed clean motion {motion['motion']}",
        )
    require(raw["latency_frames_50pct"] == 0, "pass-through introduced step latency")

    checks = (
        (
            rows["ema"]["macro_screen_jitter_reduction_pct"] >= 20.0,
            "EMA macro RMS reduction fell below 20%",
        ),
        (
            rows["ema"]["worst_screen_jitter_reduction_pct"] >= 0.0,
            "EMA worsened at least one noise cell",
        ),
        (
            rows["fir"]["macro_screen_jitter_reduction_pct"] >= 30.0,
            "FIR macro RMS reduction fell below 30%",
        ),
        (
            rows["fir"]["worst_screen_jitter_reduction_pct"] >= 5.0,
            "FIR worst-cell RMS reduction fell below 5%",
        ),
        (
            rows["oneeuro"]["macro_screen_jitter_reduction_pct"] >= 5.0,
            "One Euro macro RMS reduction fell below 5%",
        ),
        (
            rows["kalman"]["macro_screen_jitter_reduction_pct"] >= 5.0,
            "Kalman macro RMS reduction fell below 5%",
        ),
        (
            rows["kalman"]["worst_screen_jitter_reduction_pct"] >= -5.0,
            "Kalman worst-cell RMS reduction fell below -5%",
        ),
        (
            abs(rows["fir"]["latency_frames_50pct"] - rows["fir"]["designed_group_delay_frames"])
            <= 1,
            "FIR step latency differs from its designed group delay",
        ),
        (
            all(
                abs(motion["screen_age_frames"] - rows["fir"]["designed_group_delay_frames"]) <= 1
                for motion in rows["fir"]["motions"]
            ),
            "FIR Screen Age differs from its designed group delay",
        ),
        (
            rows["ema"]["latency_frames_50pct"] <= 2,
            "EMA 50% step latency exceeded two frames",
        ),
        (
            rows["oneeuro"]["latency_frames_50pct"] <= 2,
            "One Euro 50% step latency exceeded two frames",
        ),
        (
            rows["kalman"]["latency_frames_50pct"] <= 2,
            "Kalman 50% step latency exceeded two frames",
        ),
        (
            rows["ema"]["macro_screen_motion_rmse_px_aligned"] <= 5.5,
            "EMA aligned screen error exceeded 5.5 pixels",
        ),
        (
            rows["fir"]["macro_screen_motion_rmse_px_aligned"] <= 6.5,
            "FIR aligned screen error exceeded 6.5 pixels",
        ),
        (
            rows["oneeuro"]["macro_screen_motion_rmse_px_aligned"] <= 4.0,
            "One Euro aligned screen error exceeded 4.0 pixels",
        ),
        (
            rows["kalman"]["macro_screen_motion_rmse_px_aligned"] <= 7.0,
            "Kalman aligned screen error exceeded 7.0 pixels",
        ),
    )
    if reference_thresholds:
        failures.extend(message for passed, message in checks if not passed)
    failures.extend(f"{path} is non-finite" for path in _nonfinite_numeric_paths(report))
    return failures


def _validate_cli(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.fps <= 10.0:
        parser.error("--fps must be greater than 10 for the default FIR design")
    if args.frames % _WORKLOAD_COUNT:
        parser.error(f"--frames must be divisible by {_WORKLOAD_COUNT}")
    minimum_per_workload = max(120, round(args.fps * 4))
    if args.frames < _WORKLOAD_COUNT * minimum_per_workload:
        parser.error(
            f"--frames must provide at least {minimum_per_workload} samples to each "
            f"of {_WORKLOAD_COUNT} cells"
        )
    if args.speed_samples < 1:
        parser.error("--speed-samples must be positive")
    if args.check and (
        args.frames != _DEFAULT_FRAMES
        or not math.isclose(args.fps, _DEFAULT_FPS)
        or args.seed != _DEFAULT_SEED
    ):
        parser.error(
            "--check validates only the frozen default protocol "
            f"(--frames {_DEFAULT_FRAMES}, --fps {_DEFAULT_FPS:g}, "
            f"--seed {_DEFAULT_SEED}); omit --check for exploratory runs"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic, model-free temporal smoothing benchmark."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=_DEFAULT_FRAMES,
        help=f"total samples across the {_WORKLOAD_COUNT} equal benchmark cells",
    )
    parser.add_argument("--fps", type=float, default=_DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--speed-samples",
        type=int,
        default=10_000,
        help="attitude updates per filter in the machine-dependent speed loop",
    )
    parser.add_argument("--format", choices=("table", "markdown", "json"), default="table")
    parser.add_argument("--output", help="write the report instead of stdout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate deterministic invariants for the frozen default protocol",
    )
    args = parser.parse_args(argv)
    _validate_cli(parser, args)

    report = run_benchmark(
        frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        speed_samples=args.speed_samples,
    )
    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True)
    elif args.format == "markdown":
        rendered = _markdown(report)
    else:
        rendered = _table(report)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(rendered)

    if args.check:
        failures = _self_check(report)
        if failures:
            print("\nself-check failed:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("\nself-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
