"""Model-free repeatability receipt for the complete synthetic file pipeline.

Throughput excludes neural inference, camera capture, display, audio remuxing,
and verification; it is not a backend-FPS or real-time latency claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

from frame2frame import __version__, pipeline
from frame2frame.config import FilterConfig, PipelineConfig, ScreenConfig
from frame2frame.pose.base import FaceObservation, HeadPose
from frame2frame.pose.scripted import ScriptedEstimator
from frame2frame.video import VideoReader
from scripts.make_demo import synthetic_clip

_ROOT = Path(__file__).resolve().parents[1]
_CHECK_WORKLOAD = (180, 3)
_EXTENDED_WORKLOAD = (1800, 5)
_SIZE = (320, 180)
_FPS = 30.0


def _git_state() -> dict[str, Any]:
    """Disclose whether the recorded commit fully describes the measured code."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        return {
            "git_commit": None,
            "git_worktree_dirty": None,
            "source_revision_status": "unavailable",
        }
    dirty = bool(status.stdout.strip())
    return {
        "git_commit": commit.stdout.strip(),
        "git_worktree_dirty": dirty,
        "source_revision_status": "dirty_worktree" if dirty else "clean_commit",
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    """Hash decoded pixels, not codec/container metadata."""
    digest = hashlib.blake2b(digest_size=32)
    frames = 0
    with VideoReader(path) as reader:
        for frame in reader:
            digest.update(np.asarray(frame.shape, dtype=np.int64).tobytes())
            digest.update(frame.tobytes())
            frames += 1
    return {"frames": frames, "decoded_blake2b": digest.hexdigest()}


def _gap(frames: int, fps: float) -> tuple[int, int]:
    """A gap just longer than the configured 0.5-second reset."""
    start = frames // 3
    return start, min(frames, start + math.ceil(0.5 * fps) + 2)


def _visible(index: int, frames: int, fps: float) -> bool:
    start, end = _gap(frames, fps)
    return not start <= index < end


def _observations(
    frames: int,
    size: tuple[int, int],
    fps: float,
) -> Callable[[int, np.ndarray], FaceObservation | None]:
    width, height = size
    face_size = min(size) * 0.16

    def at(index: int, _frame: np.ndarray) -> FaceObservation | None:
        if not _visible(index, frames, fps):
            return None
        phase = index / frames
        cx = width * (0.5 + 0.12 * math.sin(2 * math.pi * phase))
        cy = height * (0.5 + 0.05 * math.sin(4 * math.pi * phase))
        pose = HeadPose(
            yaw=45.0 * math.sin(2 * math.pi * phase),
            pitch=30.0 * math.sin(4 * math.pi * phase),
            roll=30.0 * math.cos(4 * math.pi * phase),
        )
        return FaceObservation(
            pose,
            (cx, cy),
            face_size,
            (cx - face_size, cy - face_size, cx + face_size, cy + face_size),
        )

    return at


def _config(source: Path, output: Path) -> PipelineConfig:
    return PipelineConfig(
        input=str(source),
        output=str(output),
        filter=FilterConfig(kind="fir"),
        screen=ScreenConfig(border_thickness=2),
        draw_axis=True,
        draw_bbox=True,
        draw_screen=True,
        plot_path=None,
        dropout_hold_seconds=0.2,
        dropout_reset_seconds=0.5,
    )


def _run(source: Path, output: Path, frames: int, fps: float, size: tuple[int, int]):
    estimator = ScriptedEstimator(_observations(frames, size, fps))
    started = time.perf_counter()
    summary = pipeline.run(_config(source, output), estimator=estimator)
    elapsed = time.perf_counter() - started
    return {
        "pipeline_seconds": elapsed,
        "pipeline_frames_s": frames / elapsed,
        "summary_frames": summary.frames,
        "summary_faces": summary.faces,
        "output": _fingerprint(output),
    }


def _cycles(
    source: Path,
    directory: Path,
    frames: int,
    runs: int,
    fps: float,
    size: tuple[int, int],
) -> tuple[list[dict[str, Any]], str]:
    """Run the same complete pipeline repeatedly and retain only receipts."""
    receipts = []
    for index in range(runs):
        output = directory / f"run-{index:02d}.mp4"
        receipt = _run(source, output, frames, fps, size)
        output.unlink()
        receipt["run"] = index + 1
        receipts.append(receipt)
    reference = receipts[0]["output"]["decoded_blake2b"]
    return receipts, reference


def _aggregate(
    receipts: list[dict[str, Any]],
    reference: str,
) -> dict[str, Any]:
    throughputs = [receipt["pipeline_frames_s"] for receipt in receipts]
    return {
        "pipeline_frames_s_mean": sum(throughputs) / len(throughputs),
        "reference_output_blake2b": reference,
        "output_digests_stable": all(
            receipt["output"]["decoded_blake2b"] == reference for receipt in receipts
        ),
    }


def _self_check(report: dict[str, Any]) -> list[str]:
    method = report["method"]
    aggregate = report["aggregate"]
    frames = method["source_frames_decoded"]
    failures = []
    if frames != method["frames_requested"]:
        failures.append("synthetic source did not decode to the requested frame count")
    if not method["dropout_schedule_exercised"]:
        failures.append("scripted dropout schedule was not fully exercised")
    for receipt in report["runs"]:
        if receipt["summary_frames"] != frames or receipt["output"]["frames"] != frames:
            failures.append(f"run {receipt['run']}: frame conservation failed")
        if receipt["summary_faces"] != method["expected_face_observations"]:
            failures.append(f"run {receipt['run']}: scripted face count drifted")
        if receipt["output"]["decoded_blake2b"] != aggregate["reference_output_blake2b"]:
            failures.append(f"run {receipt['run']}: decoded output digest drifted")
    if not aggregate["output_digests_stable"]:
        failures.append("repeated runs produced different decoded output pixels")
    return failures


def _validate(frames: int, runs: int) -> None:
    if frames < math.ceil(2 * _FPS):
        raise ValueError("frames must cover at least two seconds")
    if runs < 1:
        raise ValueError("runs must be positive")


def run_benchmark(
    *,
    frames: int = _CHECK_WORKLOAD[0],
    runs: int = _CHECK_WORKLOAD[1],
) -> dict[str, Any]:
    """Return one machine-readable reliability receipt."""
    _validate(frames, runs)
    with tempfile.TemporaryDirectory(prefix="frame2frame-pipeline-") as name:
        directory = Path(name)
        source = directory / "source.mp4"
        synthetic_clip(source, frames=frames, size=_SIZE, fps=_FPS)
        source_before = _fingerprint(source)
        decoded_frames = source_before["frames"]
        receipts, reference = _cycles(source, directory, decoded_frames, runs, _FPS, _SIZE)

    gap_start, gap_end = _gap(decoded_frames, _FPS)
    gap_duration = (gap_end - gap_start) / _FPS
    expected_faces = sum(_visible(index, decoded_frames, _FPS) for index in range(decoded_frames))
    report = {
        "schema_version": 1,
        "status": "unchecked",
        "method": {
            "frames_requested": frames,
            "source_frames_decoded": decoded_frames,
            "expected_face_observations": expected_faces,
            "fps": _FPS,
            "resolution": list(_SIZE),
            "measured_runs": runs,
            "dropout_frame_range": [gap_start, gap_end],
            "dropout_duration_seconds": gap_duration,
            "dropout_reset_seconds": 0.5,
            "observations_after_dropout": max(0, decoded_frames - gap_end),
            "dropout_schedule_exercised": (
                0 < gap_start < gap_end < decoded_frames and gap_duration > 0.5
            ),
        },
        "environment": {
            **_git_state(),
            "project_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
        },
        "runs": receipts,
        "aggregate": _aggregate(receipts, reference),
        "failures": [],
        "evidence_boundaries": [
            "ScriptedEstimator performs no neural inference; throughput is not backend FPS.",
            "File processing is unpaced; frames/s is not camera-to-display latency.",
            "The schedule proves a reset-length gap and later observations, not reset mechanics.",
            "This regression receipt does not establish pose accuracy or production SLOs.",
        ],
    }
    report["failures"] = _self_check(report)
    report["status"] = "pass" if not report["failures"] else "fail"
    return report


def _markdown(report: dict[str, Any]) -> str:
    method, aggregate = report["method"], report["aggregate"]
    lines = [
        "# Pipeline reliability receipt",
        "",
        f"**Status:** `{report['status'].upper()}`  ",
        f"**Commit:** `{report['environment']['git_commit']}` "
        f"(`{report['environment']['source_revision_status']}`)  ",
        "> Synthetic file pipeline only; not neural-backend FPS or camera-to-display latency.",
        "",
        "| Run | Frames in/summary/out | Faces | Seconds | Pipeline frames/s | Digest |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for receipt in report["runs"]:
        lines.append(
            f"| {receipt['run']} | {method['source_frames_decoded']}/"
            f"{receipt['summary_frames']}/{receipt['output']['frames']} | "
            f"{receipt['summary_faces']} | {receipt['pipeline_seconds']:.3f} | "
            f"{receipt['pipeline_frames_s']:.1f} | "
            f"`{receipt['output']['decoded_blake2b'][:12]}` |"
        )
    lines.extend(
        [
            "",
            f"- Stable decoded output: {aggregate['output_digests_stable']}.",
            f"- Scripted dropout: {method['dropout_duration_seconds']:.3f}s, followed by "
            f"{method['observations_after_dropout']} scheduled observations.",
            f"- Mean pipeline throughput: {aggregate['pipeline_frames_s_mean']:.1f} frames/s "
            "(no pass/fail floor).",
            "",
            "## Evidence boundaries",
            "",
            *(f"- {boundary}" for boundary in report["evidence_boundaries"]),
        ]
    )
    if report["failures"]:
        lines.extend(["", "## Failures", "", *(f"- {item}" for item in report["failures"])])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="repeat a 1,800-frame source five times",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="write the receipt instead of stdout")
    parser.add_argument("--check", action="store_true", help="fail on a contract regression")
    args = parser.parse_args(argv)
    defaults = _EXTENDED_WORKLOAD if args.extended else _CHECK_WORKLOAD
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
