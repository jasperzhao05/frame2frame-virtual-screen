"""Model-free benchmark receipt for the dynamic screen-content fast path.

The default workload is intentionally small enough for CI.  Timing values are
reported, never used as pass/fail gates: the self-check covers structural
contracts that should remain portable across Python, OpenCV, and CPU versions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import sys
import time
import weakref
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frame2frame import ContentRequest, LatestFrameSource, render
from frame2frame._textures import prepare_texture

_DEFAULT_FRAMES = 24
_DEFAULT_RUNS = 2
_EXTENDED_FRAMES = 300
_EXTENDED_RUNS = 3
_FPS = 30.0
_FRAME_SIZE = (320, 180)
_CONTENT_SIZE = (128, 96)
_QUAD = np.float32([[48.25, 35.5], [273.75, 31.25], [268.5, 149.75], [43.0, 153.5]])


def _validate_workload(frames: object, runs: object) -> tuple[int, int]:
    for name, value in (("frames", frames), ("runs", runs)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return frames, runs


def _frame(index: int, size: tuple[int, int]) -> np.ndarray:
    """Return one deterministic contiguous BGR frame with an exact index marker."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.empty((height, width, 3), np.uint8)
    image[..., 0] = (xx + 17 * index) % 256
    image[..., 1] = (2 * yy + 29 * index) % 256
    image[..., 2] = (xx + yy + 43 * index) % 256
    image[0, 0] = (index & 0xFF, (index >> 8) & 0xFF, (index >> 16) & 0xFF)
    return np.ascontiguousarray(image)


def _frame_index(frame: np.ndarray) -> int:
    blue, green, red = (int(value) for value in frame[0, 0])
    return blue | (green << 8) | (red << 16)


def _percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile with no optional statistics dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _structural_contracts() -> dict[str, bool]:
    """Check latest-only storage and the opaque BGR preparation fast path."""
    source = LatestFrameSource()
    request = ContentRequest(0, 0.0, True)
    first = _frame(1, (8, 4))
    second = _frame(2, (8, 4))
    latest = _frame(3, (8, 4))
    first_ref = weakref.ref(first)
    second_ref = weakref.ref(second)

    source.publish(first, copy=False)
    source.publish(second, copy=False)
    source.publish(latest, copy=False)
    sampled = source.frame_at(request)

    del first, second
    gc.collect()
    prepared = prepare_texture(sampled)
    return {
        "latest_wins": sampled is latest and _frame_index(sampled) == 3,
        # A source retaining a queue would keep at least one overwritten array alive.
        "capacity_one_by_design": first_ref() is None and second_ref() is None,
        "zero_copy_prepared_bgr": np.shares_memory(prepared.bgr, sampled),
        "opaque_alpha_fast_path": prepared.alpha is None,
    }


def _run_once(frames: int) -> dict[str, Any]:
    source = LatestFrameSource()
    sample_ms: list[float] = []
    prepare_ms: list[float] = []
    composite_ms: list[float] = []
    mapping = hashlib.blake2b(digest_size=32)
    output = hashlib.blake2b(digest_size=32)
    mapping_exact = True
    zero_copy_every_frame = True

    for index in range(frames):
        content = _frame(index, _CONTENT_SIZE)
        source.publish(content, copy=False)
        request = ContentRequest(index, index / _FPS, True)

        started = time.perf_counter_ns()
        sampled = source.frame_at(request)
        sample_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if sampled is None:
            raise RuntimeError("latest source unexpectedly returned no frame")

        observed_index = _frame_index(sampled)
        mapping_exact = mapping_exact and observed_index == index
        mapping.update(struct.pack("<QQ", index, observed_index))

        started = time.perf_counter_ns()
        prepared = prepare_texture(sampled)
        prepare_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        zero_copy_every_frame = zero_copy_every_frame and np.shares_memory(prepared.bgr, sampled)

        target = _frame(index + 10_000, _FRAME_SIZE)
        started = time.perf_counter_ns()
        render._paste_content(
            target,
            prepared,
            _QUAD,
            0.8,
            fit="contain",
            target_aspect=16.0 / 9.0,
        )
        composite_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        output.update(np.asarray(target.shape, dtype=np.int64).tobytes())
        output.update(target.tobytes())

    return {
        "frames": frames,
        "mapping_exact": mapping_exact,
        "zero_copy_every_frame": zero_copy_every_frame,
        "mapping_blake2b": mapping.hexdigest(),
        "output_blake2b": output.hexdigest(),
        "timings": {
            "sample": _timing_summary(sample_ms),
            "prepare": _timing_summary(prepare_ms),
            "composite": _timing_summary(composite_ms),
        },
        "_timing_samples": {
            "sample": sample_ms,
            "prepare": prepare_ms,
            "composite": composite_ms,
        },
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    reference_mapping = runs[0]["mapping_blake2b"]
    reference_output = runs[0]["output_blake2b"]
    timings = {
        stage: [value for receipt in runs for value in receipt["_timing_samples"][stage]]
        for stage in ("sample", "prepare", "composite")
    }
    return {
        "frame_mapping_stable": all(
            receipt["mapping_blake2b"] == reference_mapping for receipt in runs
        ),
        "output_digest_stable": all(
            receipt["output_blake2b"] == reference_output for receipt in runs
        ),
        "reference_mapping_blake2b": reference_mapping,
        "reference_output_blake2b": reference_output,
        "timings": {stage: _timing_summary(values) for stage, values in timings.items()},
    }


def _self_check(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, passed in report["contracts"].items():
        if not passed:
            failures.append(f"structural contract failed: {name}")
    for receipt in report["runs"]:
        if receipt["frames"] != report["method"]["frames_per_run"]:
            failures.append(f"run {receipt['run']}: frame count drifted")
        if not receipt["mapping_exact"]:
            failures.append(f"run {receipt['run']}: frame mapping drifted")
        if not receipt["zero_copy_every_frame"]:
            failures.append(f"run {receipt['run']}: BGR preparation copied a frame")
    if not report["aggregate"]["frame_mapping_stable"]:
        failures.append("frame mapping digest changed between repeated runs")
    if not report["aggregate"]["output_digest_stable"]:
        failures.append("composited output digest changed between repeated runs")
    return failures


def run_benchmark(
    *,
    frames: int = _DEFAULT_FRAMES,
    runs: int = _DEFAULT_RUNS,
) -> dict[str, Any]:
    """Return one deterministic dynamic-content benchmark receipt."""
    frames, runs = _validate_workload(frames, runs)
    measured = [_run_once(frames) for _ in range(runs)]
    aggregate = _aggregate(measured)
    receipts = []
    for run_index, receipt in enumerate(measured):
        receipt.pop("_timing_samples")
        receipt["run"] = run_index + 1
        receipts.append(receipt)

    report = {
        "schema_version": 1,
        "status": "unchecked",
        "method": {
            "frames_per_run": frames,
            "runs": runs,
            "fps": _FPS,
            "frame_size": list(_FRAME_SIZE),
            "content_size": list(_CONTENT_SIZE),
            "content_fit": "contain",
            "model_or_codec_used": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "contracts": _structural_contracts(),
        "runs": receipts,
        "aggregate": aggregate,
        "failures": [],
        "evidence_boundaries": [
            "Timing values are measurements, not portable pass/fail thresholds.",
            "The workload excludes video codecs, pose inference, display, and output encoding.",
            "LatestFrameSource uses copy=False here; the benchmark never mutates published frames.",
            "The receipt checks frame selection and compositing determinism, not visual quality.",
        ],
    }
    report["failures"] = _self_check(report)
    report["status"] = "pass" if not report["failures"] else "fail"
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dynamic-content benchmark receipt",
        "",
        f"**Status:** `{report['status'].upper()}`  ",
        "> Model- and codec-free; measured milliseconds are not CI timing gates.",
        "",
        "| Stage | p50 (ms) | p95 (ms) | Samples |",
        "|---|---:|---:|---:|",
    ]
    for stage in ("sample", "prepare", "composite"):
        timing = report["aggregate"]["timings"][stage]
        lines.append(
            f"| {stage} | {timing['p50_ms']:.4f} | {timing['p95_ms']:.4f} | {timing['samples']} |"
        )
    lines.extend(["", "## Structural contracts", ""])
    lines.extend(
        f"- {name}: `{str(passed).lower()}`" for name, passed in report["contracts"].items()
    )
    lines.extend(
        [
            f"- frame_mapping_stable: `{str(report['aggregate']['frame_mapping_stable']).lower()}`",
            f"- output_digest_stable: `{str(report['aggregate']['output_digest_stable']).lower()}`",
            "",
            "## Evidence boundaries",
            "",
            *(f"- {item}" for item in report["evidence_boundaries"]),
        ]
    )
    if report["failures"]:
        lines.extend(["", "## Failures", "", *(f"- {item}" for item in report["failures"])])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    defaults = (
        (_EXTENDED_FRAMES, _EXTENDED_RUNS) if args.extended else (_DEFAULT_FRAMES, _DEFAULT_RUNS)
    )
    try:
        report = run_benchmark(
            frames=defaults[0] if args.frames is None else args.frames,
            runs=defaults[1] if args.runs is None else args.runs,
        )
    except ValueError as error:
        parser.error(str(error))

    rendered = (
        json.dumps(report, indent=2, sort_keys=True) if args.format == "json" else _markdown(report)
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {output}")
    else:
        print(rendered)
    if args.check and report["failures"]:
        for failure in report["failures"]:
            print(f"- {failure}", file=sys.stderr)
        return 1
    if args.check:
        print("\nself-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
