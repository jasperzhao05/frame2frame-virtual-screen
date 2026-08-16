"""Create an evidence-bounded receipt for one real-video pipeline run.

The receipt records content hashes, exact configuration, decoded media facts,
fresh-detection coverage, and timing observed during the run.  It deliberately
does not turn those operational measurements into pose-accuracy, perceptual-
quality, latency-SLO, or production-readiness claims.

Run from a source checkout:

    python -m scripts.validate_real_video \
      --input examples/inputs/head-pose-face-detection-female.mp4 \
      --output examples/outputs/head-pose-face-detection-female.validated.mp4 \
      --receipt output/real-video-validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import scipy

from frame2frame import PipelineConfig, RunSummary, __version__, run
from frame2frame._downloads import sha256_file
from frame2frame.config import FilterConfig
from frame2frame.video import VideoReader
from scripts.fetch_examples import BASE, CLIPS, LICENSE, REPOSITORY

SCHEMA_VERSION = 1
_SOURCE_LICENSE = f"Creative Commons Attribution 4.0 International ({LICENSE})"
_SOURCE_ATTRIBUTION = "Intel IoT DevKit sample-videos contributors"


@dataclass(frozen=True)
class MediaFacts:
    path: str
    sha256: str
    size_bytes: int
    decoded_frames: int
    fps: float
    width_px: int
    height_px: int
    decoded_duration_seconds: float


@dataclass(frozen=True)
class SourceProvenance:
    uri: str
    license: str
    attribution: str
    registry_match: str | None


def probe_video(path: Path) -> MediaFacts:
    """Decode every frame and report facts used by the conservation checks."""
    if not path.is_file():
        raise FileNotFoundError(f"video does not exist: {path}")
    with VideoReader(path) as reader:
        frames = sum(1 for _ in reader)
        fps = float(reader.fps)
        width, height = reader.size
    return MediaFacts(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        decoded_frames=frames,
        fps=fps,
        width_px=width,
        height_px=height,
        decoded_duration_seconds=frames / fps,
    )


def resolve_provenance(
    path: Path,
    digest: str,
    *,
    source_uri: str | None,
    source_license: str | None,
    source_attribution: str | None,
) -> SourceProvenance:
    """Resolve pinned examples or require a complete caller-supplied source note."""
    for name, clip in CLIPS.items():
        if digest == clip.sha256:
            return SourceProvenance(
                uri=f"{BASE}/{name}",
                license=_SOURCE_LICENSE,
                attribution=_SOURCE_ATTRIBUTION,
                registry_match=f"{REPOSITORY}::{name}",
            )

    if (
        source_uri is None
        or not source_uri.strip()
        or source_license is None
        or not source_license.strip()
        or source_attribution is None
        or not source_attribution.strip()
    ):
        raise ValueError(
            f"{path} is not a pinned example; provide --source-uri, "
            "--source-license, and --source-attribution"
        )
    return SourceProvenance(
        uri=source_uri.strip(),
        license=source_license.strip(),
        attribution=source_attribution.strip(),
        registry_match=None,
    )


def _git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}
    return {"commit": commit, "worktree_dirty": dirty}


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "mediapipe": _installed_version("mediapipe"),
    }


def _revision_status(revision: dict[str, object]) -> str:
    commit = revision.get("commit")
    dirty = revision.get("worktree_dirty")
    if not isinstance(commit, str) or not commit or not isinstance(dirty, bool):
        return "unavailable"
    return "dirty_worktree" if dirty else "clean_commit"


def _same_media_rate(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-3, abs_tol=0.02)


def _duration_tolerance(first: MediaFacts, second: MediaFacts) -> float:
    return max(1.0 / first.fps, 1.0 / second.fps) + 1e-6


def _check(name: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def build_receipt(
    *,
    config: PipelineConfig,
    provenance: SourceProvenance,
    input_media: MediaFacts,
    output_media: MediaFacts,
    summary: RunSummary,
    pipeline_wall_seconds: float,
    command: list[str],
    project_state: dict[str, object] | None = None,
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the stable schema from separately supplied measurements."""
    revision = project_state if project_state is not None else _git_state()
    detection_rate = 100.0 * summary.faces / summary.frames if summary.frames else 0.0
    wall_rate = summary.frames / pipeline_wall_seconds if pipeline_wall_seconds > 0 else 0.0
    duration_delta = abs(
        input_media.decoded_duration_seconds - output_media.decoded_duration_seconds
    )
    checks = [
        _check(
            "pipeline processed every decoded input frame",
            summary.frames == input_media.decoded_frames,
            {
                "pipeline_frames": summary.frames,
                "input_decoded_frames": input_media.decoded_frames,
            },
        ),
        _check(
            "pipeline used the separately probed input frame rate",
            _same_media_rate(input_media.fps, summary.fps),
            {"input_fps": input_media.fps, "pipeline_fps": summary.fps},
        ),
        _check(
            "output preserved the processed frame count",
            output_media.decoded_frames == summary.frames,
            {
                "output_decoded_frames": output_media.decoded_frames,
                "pipeline_frames": summary.frames,
            },
        ),
        _check(
            "output preserved decoded dimensions",
            (output_media.width_px, output_media.height_px)
            == (input_media.width_px, input_media.height_px),
            {
                "input_width_px": input_media.width_px,
                "input_height_px": input_media.height_px,
                "output_width_px": output_media.width_px,
                "output_height_px": output_media.height_px,
            },
        ),
        _check(
            "output preserved effective frame rate",
            _same_media_rate(input_media.fps, output_media.fps),
            {"input_fps": input_media.fps, "output_fps": output_media.fps},
        ),
        _check(
            "output duration stayed within one frame",
            duration_delta <= _duration_tolerance(input_media, output_media),
            {
                "duration_delta_seconds": duration_delta,
                "tolerance_seconds": _duration_tolerance(input_media, output_media),
            },
        ),
        _check(
            "backend produced at least one fresh detection",
            summary.faces > 0,
            {"fresh_detection_frames": summary.faces},
        ),
        _check(
            "fresh-detection count was within processed-frame bounds",
            0 <= summary.faces <= summary.frames,
            {"fresh_detection_frames": summary.faces, "pipeline_frames": summary.frames},
        ),
        _check(
            "recorded timings were finite and non-negative",
            (
                math.isfinite(summary.mean_inference_ms)
                and summary.mean_inference_ms >= 0
                and math.isfinite(pipeline_wall_seconds)
                and pipeline_wall_seconds > 0
            ),
            {
                "mean_estimator_inference_ms": summary.mean_inference_ms,
                "pipeline_wall_seconds": pipeline_wall_seconds,
            },
        ),
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": "real-video-operational-validation",
        "status": "pass" if passed else "fail",
        "scope": {
            "purpose": (
                "Reproduce one real-footage decode, pose-estimation, smoothing, "
                "render, and encode run."
            ),
            "measures": [
                "content identity and provenance",
                "decoded media structure and frame conservation",
                "fresh backend detections per processed frame",
                "mean estimator-call time and whole-pipeline wall time on this machine",
            ],
            "does_not_measure": [
                "ground-truth head-pose or detection accuracy",
                "perceptual stability or visual quality",
                "capture-to-display latency or a latency service level objective",
                "performance portability, reliability, or production readiness",
            ],
        },
        "project": {
            "version": __version__,
            **revision,
            "source_revision_status": _revision_status(revision),
        },
        "command": command,
        "source": asdict(provenance),
        "configuration": asdict(config),
        "input": asdict(input_media),
        "run": {
            "frames_processed": summary.frames,
            "fresh_detection_frames": summary.faces,
            "fresh_detection_frame_rate_pct": detection_rate,
            "source_fps": summary.fps,
            "mean_estimator_inference_ms": summary.mean_inference_ms,
            "pipeline_wall_seconds": pipeline_wall_seconds,
            "processed_frames_per_wall_second": wall_rate,
            "audio_remuxed": summary.audio_remuxed,
        },
        "output": asdict(output_media),
        "checks": checks,
        "environment": environment if environment is not None else _environment(),
    }


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.tmp")
    try:
        staged.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _distinct_receipt_path(input_path: Path, output_path: Path, receipt_path: Path) -> None:
    resolved = [path.expanduser().resolve() for path in (input_path, output_path, receipt_path)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input, output, and receipt must use three different paths")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the primary MediaPipe + FIR real-video path."
    )
    parser.add_argument("--input", required=True, help="real video to process")
    parser.add_argument(
        "--output",
        default="output/real-video-validation.mp4",
        help="processed video destination",
    )
    parser.add_argument(
        "--receipt",
        default="output/real-video-validation.json",
        help="JSON receipt destination",
    )
    provenance = parser.add_argument_group("provenance for inputs outside the pinned examples")
    provenance.add_argument("--source-uri", help="public URL or stable source identifier")
    provenance.add_argument("--source-license", help="license name and, when available, URL")
    provenance.add_argument("--source-attribution", help="person or organization to credit")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _parser()
    args = parser.parse_args(raw_argv)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    receipt_path = Path(args.receipt).expanduser()

    try:
        _distinct_receipt_path(input_path, output_path, receipt_path)
        input_media = probe_video(input_path)
        provenance = resolve_provenance(
            input_path,
            input_media.sha256,
            source_uri=args.source_uri,
            source_license=args.source_license,
            source_attribution=args.source_attribution,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    config = PipelineConfig(
        input=str(input_path),
        output=str(output_path),
        backend="mediapipe",
        filter=FilterConfig(kind="fir"),
        plot_path=None,
        preserve_audio=False,
    )
    started = time.perf_counter()
    summary = run(config)
    wall_seconds = time.perf_counter() - started
    output_media = probe_video(output_path)
    receipt = build_receipt(
        config=config,
        provenance=provenance,
        input_media=input_media,
        output_media=output_media,
        summary=summary,
        pipeline_wall_seconds=wall_seconds,
        command=["python", "-m", "scripts.validate_real_video", *raw_argv],
    )
    _write_receipt(receipt_path, receipt)

    print(f"wrote {receipt_path}")
    print(
        f"validation {str(receipt['status']).upper()}: "
        f"{summary.faces}/{summary.frames} frames had a fresh detection; "
        f"mean estimator call {summary.mean_inference_ms:.1f} ms"
    )
    print("scope: operational receipt only; not pose-accuracy or production evidence")
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
