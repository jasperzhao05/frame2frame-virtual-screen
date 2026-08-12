"""Internal atomic media publishing and optional source-audio remuxing."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, Union, cast


@contextmanager
def _temporary_sibling(output: Path, purpose: str) -> Iterator[Path]:
    """Yield a same-container temporary file and always remove leftovers."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.stem}.{purpose}-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        yield staged
    finally:
        staged.unlink(missing_ok=True)


def _install_nonempty(staged: Path, output: Path, *, error: str) -> str:
    if not staged.is_file() or staged.stat().st_size == 0:
        raise RuntimeError(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, output)
    return str(output)


@contextmanager
def staged_video_path(output_path: str | os.PathLike[str]) -> Iterator[str]:
    """Yield a temporary sibling path suitable for OpenCV video encoding.

    The original suffix selects the right OpenCV container. The caller decides
    when to install the completed file; this context only cleans up leftovers.
    """
    output = Path(output_path).expanduser()
    if not output.suffix:
        raise ValueError("output filename must include a container suffix")
    with _temporary_sibling(output, "video") as staged:
        yield str(staged)


def install_video(
    processed_video: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> str:
    """Atomically install a completed, non-empty staged video."""
    staged = Path(processed_video).expanduser()
    output = Path(output_path).expanduser()
    return _install_nonempty(staged, output, error="processed video is missing or empty")


def resolve_ffmpeg(executable: str | os.PathLike[str] | None = None) -> str:
    """Resolve ffmpeg before expensive processing starts."""
    resolved = str(executable) if executable else shutil.which("ffmpeg")
    if not resolved:
        raise RuntimeError("preserve_audio requires ffmpeg on PATH")
    return resolved


def _positive_duration(value: object) -> float | None:
    """Return a finite positive duration while preserving the prior coercion policy."""
    try:
        duration = float(cast(Union[str, SupportsFloat, SupportsIndex], value))
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


@dataclass(frozen=True)
class _MuxRequest:
    processed: Path
    source: Path
    output: Path
    duration: float
    executable: str


def _validated_mux_request(
    processed_video: str | os.PathLike[str],
    source_video: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    duration_seconds: object,
    ffmpeg: str | os.PathLike[str] | None,
) -> _MuxRequest:
    """Validate all mux inputs in the same order as the public operation."""
    processed = Path(processed_video).expanduser()
    source = Path(source_video).expanduser()
    output = Path(output_path).expanduser()
    if not processed.is_file():
        raise FileNotFoundError(f"processed video does not exist: {processed}")
    if not source.is_file():
        raise FileNotFoundError(f"source video does not exist: {source}")
    if not output.suffix:
        raise ValueError("audio remux output must have a filename suffix")
    duration = _positive_duration(duration_seconds)
    if duration is None:
        raise ValueError("audio remux duration must be a finite positive number")
    return _MuxRequest(processed, source, output, duration, resolve_ffmpeg(ffmpeg))


def _ffmpeg_mux_command(request: _MuxRequest, staged: Path) -> list[str]:
    """Build the exact stream-copy command for one validated request."""
    return [
        request.executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(request.processed),
        "-i",
        str(request.source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-t",
        f"{request.duration:.9f}",
        str(staged),
    ]


def _run_mux_command(command: list[str]) -> None:
    """Run ffmpeg and translate its concise stderr into the public error."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio remux failed: {detail}")


def mux_audio(
    processed_video: str | os.PathLike[str],
    source_video: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    duration_seconds: object,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> str:
    """Copy source audio into processed video and atomically install the result.

    The existing output is never touched unless ffmpeg exits successfully and
    produces a non-empty file. ``1:a:0?`` deliberately permits silent inputs;
    in that case this still validates and installs the processed video. The
    explicit duration follows the processed frame stream: short audio must not
    truncate video, and long audio must not extend it.
    """
    request = _validated_mux_request(
        processed_video,
        source_video,
        output_path,
        duration_seconds=duration_seconds,
        ffmpeg=ffmpeg,
    )
    with _temporary_sibling(request.output, "mux") as staged:
        _run_mux_command(_ffmpeg_mux_command(request, staged))
        return _install_nonempty(
            staged,
            request.output,
            error="ffmpeg audio remux produced no output",
        )
