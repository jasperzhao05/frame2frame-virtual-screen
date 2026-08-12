"""Deterministic, model-free benchmark for frame2frame's temporal filters.

The benchmark separates observation noise from intentional head motion by
feeding the same seeded synthetic trace through two copies of each filter:
one receives the clean trace and one receives the noisy trace.  Their
difference is the residual jitter.  A separate step response measures 50%
response latency, and a tight update loop reports filter throughput.

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
from pathlib import Path
from typing import TypedDict

import numpy as np

from frame2frame.config import FilterConfig
from frame2frame.filters import create_filter

_FILTERS = ("none", "fir", "oneeuro")
_AXES = ("yaw", "pitch", "roll")


class BenchmarkRow(TypedDict):
    filter: str
    jitter_rms_deg: float
    jitter_reduction_pct: float
    axis_jitter_rms_deg: dict[str, float]
    latency_frames_50pct: int
    latency_ms_50pct: float
    designed_group_delay_frames: int
    motion_rmse_deg_aligned: float
    throughput_observations_s: float


class BenchmarkMethod(TypedDict):
    frames: int
    fps: float
    seed: int
    speed_samples: int
    input_jitter_rms_deg: float
    latency_definition: str
    jitter_definition: str


class BenchmarkEnvironment(TypedDict):
    python: str
    platform: str
    numpy: str


class BenchmarkReport(TypedDict):
    schema_version: int
    method: BenchmarkMethod
    environment: BenchmarkEnvironment
    results: list[BenchmarkRow]


def _trace(frames: int, fps: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return clean and noisy yaw/pitch/roll traces in degrees."""
    t = np.arange(frames, dtype=np.float64) / fps
    clean = np.column_stack(
        (
            20.0 * np.sin(2.0 * np.pi * 0.18 * t) + 4.0 * np.sin(2.0 * np.pi * 0.55 * t + 0.4),
            10.0 * np.sin(2.0 * np.pi * 0.23 * t + 0.8),
            7.0 * np.sin(2.0 * np.pi * 0.14 * t - 0.3),
        )
    )
    rng = np.random.default_rng(seed)
    noise_std = np.array([1.25, 1.0, 1.6], dtype=np.float64)
    noise = rng.normal(0.0, noise_std, size=clean.shape)

    # A small high-frequency detector wobble makes the input more representative
    # than white noise alone while remaining exactly reproducible.
    wobble = np.column_stack(
        (
            0.35 * np.sin(2.0 * np.pi * 8.0 * t),
            0.30 * np.sin(2.0 * np.pi * 7.0 * t + 0.7),
            0.45 * np.sin(2.0 * np.pi * 9.0 * t + 1.1),
        )
    )
    return clean, clean + noise + wobble


def _apply(kind: str, values: np.ndarray, fps: float) -> tuple[np.ndarray, int]:
    filt = create_filter(fps, FilterConfig(kind=kind))
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


def _throughput(kind: str, fps: float, samples: int) -> float:
    cfg = FilterConfig(kind=kind)
    filt = create_filter(fps, cfg)
    t0 = time.perf_counter()
    checksum = 0.0
    for index in range(samples):
        phase = index * 0.017
        yaw, pitch, roll = filt.update(
            18.0 * math.sin(phase),
            9.0 * math.sin(phase * 0.7 + 0.2),
            5.0 * math.sin(phase * 0.4 - 0.3),
        )
        cx, cy, size = filt.update_position(
            320.0 + 24.0 * math.sin(phase * 0.5),
            240.0 + 12.0 * math.cos(phase * 0.6),
            72.0 + 3.0 * math.sin(phase * 0.3),
        )
        checksum += yaw + pitch + roll + cx + cy + size
    elapsed = time.perf_counter() - t0
    # Keep the loop observable without printing a meaningless implementation detail.
    if not math.isfinite(checksum):
        raise RuntimeError("non-finite filter output")
    return samples / elapsed


def run_benchmark(
    *,
    frames: int = 1800,
    fps: float = 30.0,
    seed: int = 20260730,
    speed_samples: int = 10_000,
) -> BenchmarkReport:
    clean, noisy = _trace(frames, fps, seed)
    baseline_jitter = float(np.sqrt(np.mean(np.square(noisy - clean))))
    results: list[BenchmarkRow] = []

    for kind in _FILTERS:
        clean_out, group_delay = _apply(kind, clean, fps)
        noisy_out, _ = _apply(kind, noisy, fps)
        latency, designed_delay = _step_latency(kind, fps)
        trim = max(round(fps), 2 * group_delay)
        residual = noisy_out[trim:] - clean_out[trim:]
        comparable_input = noisy[trim:] - clean[trim:]
        comparable_input_jitter = float(np.sqrt(np.mean(np.square(comparable_input))))
        axis_jitter = np.sqrt(np.mean(np.square(residual), axis=0))
        jitter = float(np.sqrt(np.mean(np.square(residual))))
        results.append(
            {
                "filter": kind,
                "jitter_rms_deg": jitter,
                "jitter_reduction_pct": 100.0 * (1.0 - jitter / comparable_input_jitter),
                "axis_jitter_rms_deg": {
                    axis: float(value) for axis, value in zip(_AXES, axis_jitter)
                },
                "latency_frames_50pct": latency,
                "latency_ms_50pct": latency / fps * 1000.0,
                "designed_group_delay_frames": designed_delay,
                "motion_rmse_deg_aligned": _aligned_rmse(clean_out, clean, latency, trim),
                "throughput_observations_s": _throughput(kind, fps, speed_samples),
            }
        )

    return {
        "schema_version": 2,
        "method": {
            "frames": frames,
            "fps": fps,
            "seed": seed,
            "speed_samples": speed_samples,
            "input_jitter_rms_deg": baseline_jitter,
            "latency_definition": "first output crossing of 50% of a 20-degree yaw step",
            "jitter_definition": (
                "RMS difference between noisy-input and clean-input filter outputs"
            ),
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
        "filter    jitter RMS   reduction   latency (50%)   aligned RMSE   throughput",
        "--------  -----------  ----------  --------------  -------------  ----------",
    ]
    for row in report["results"]:
        lines.append(
            f"{row['filter']:<8}  "
            f"{row['jitter_rms_deg']:>8.3f}°  "
            f"{row['jitter_reduction_pct']:>8.1f}%  "
            f"{row['latency_frames_50pct']:>4d} fr / "
            f"{row['latency_ms_50pct']:>6.1f} ms  "
            f"{row['motion_rmse_deg_aligned']:>8.3f}°  "
            f"{row['throughput_observations_s'] / 1000:>7.1f}k/s"
        )
    method = report["method"]
    lines.extend(
        (
            "",
            f"seed={method['seed']}  frames={method['frames']}  "
            f"fps={method['fps']:.1f}  input jitter={method['input_jitter_rms_deg']:.3f}°",
            "Throughput is machine-dependent; quality and latency inputs are deterministic.",
        )
    )
    return "\n".join(lines)


def _markdown(report: BenchmarkReport) -> str:
    lines = [
        "| Filter | Residual jitter RMS | Jitter reduction | 50% latency | "
        "Aligned motion RMSE | Filter throughput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| `{row['filter']}` | {row['jitter_rms_deg']:.3f}° | "
            f"{row['jitter_reduction_pct']:.1f}% | "
            f"{row['latency_frames_50pct']} frames / "
            f"{row['latency_ms_50pct']:.1f} ms | "
            f"{row['motion_rmse_deg_aligned']:.3f}° | "
            f"{row['throughput_observations_s'] / 1000:.1f}k observations/s |"
        )
    return "\n".join(lines)


def _self_check(report: BenchmarkReport) -> list[str]:
    rows = {row["filter"]: row for row in report["results"]}
    checks = (
        (
            abs(rows["none"]["jitter_reduction_pct"]) > 0.01,
            "pass-through changed the synthetic jitter",
        ),
        (
            rows["fir"]["jitter_reduction_pct"] < 70.0,
            "FIR removed less than 70% of synthetic jitter",
        ),
        (
            rows["oneeuro"]["jitter_reduction_pct"] < 20.0,
            "One Euro removed less than 20% of synthetic jitter",
        ),
        (
            abs(rows["fir"]["latency_frames_50pct"] - rows["fir"]["designed_group_delay_frames"])
            > 1,
            "FIR step latency differs from its designed group delay",
        ),
        (
            rows["fir"]["motion_rmse_deg_aligned"] > 0.15,
            "FIR aligned motion RMSE exceeded 0.15 degrees",
        ),
        (
            rows["oneeuro"]["latency_frames_50pct"] > 2,
            "One Euro 50% step latency exceeded two frames",
        ),
        (
            rows["oneeuro"]["motion_rmse_deg_aligned"] > 0.75,
            "One Euro aligned motion RMSE exceeded 0.75 degrees",
        ),
        (
            rows["none"]["latency_frames_50pct"] != 0,
            "pass-through introduced step latency",
        ),
        (
            rows["none"]["motion_rmse_deg_aligned"] > 1e-12,
            "pass-through changed the clean motion trace",
        ),
    )
    failures = [message for failed, message in checks if failed]
    for row in rows.values():
        numeric = (
            row["jitter_rms_deg"],
            row["jitter_reduction_pct"],
            row["latency_ms_50pct"],
            row["motion_rmse_deg_aligned"],
            row["throughput_observations_s"],
        )
        if not all(math.isfinite(value) for value in numeric):
            failures.append(f"{row['filter']} produced a non-finite metric")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic, model-free temporal smoothing benchmark."
    )
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--speed-samples",
        type=int,
        default=10_000,
        help="observations per filter in the machine-dependent speed loop",
    )
    parser.add_argument("--format", choices=("table", "markdown", "json"), default="table")
    parser.add_argument("--output", help="write the report instead of stdout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if deterministic quality/latency invariants regress",
    )
    args = parser.parse_args(argv)

    if args.frames < max(120, round(args.fps * 4)):
        parser.error("--frames must cover at least four seconds and 120 samples")
    if args.fps <= 10.0:
        parser.error("--fps must be greater than 10 for the default FIR design")
    if args.speed_samples < 1:
        parser.error("--speed-samples must be positive")

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
