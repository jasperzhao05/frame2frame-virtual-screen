"""Internal no-download, no-device runtime readiness checks for the CLI."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from contextlib import redirect_stderr
from dataclasses import dataclass
from importlib import import_module, metadata
from io import StringIO
from pathlib import Path

from ._downloads import sha256_file
from ._version import __version__


@dataclass(frozen=True)
class _DoctorCheck:
    name: str
    level: str
    detail: str


_REQUIRED_IMPORTS = (
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("cv2", "OpenCV"),
)
_REQUIRED_DISTRIBUTIONS = (("matplotlib", "Matplotlib"),)
_MEDIAPIPE_RANGE = ">=0.10.9,<1"


def _python_check() -> _DoctorCheck:
    version = sys.version_info
    rendered = f"{version.major}.{version.minor}.{version.micro}"
    if (3, 9) <= version[:2] < (3, 14):
        return _DoctorCheck("python", "ok", f"{rendered} is supported")
    return _DoctorCheck("python", "error", f"{rendered} is outside the supported 3.9-3.13 range")


def _dependency_check(module_name: str, label: str) -> _DoctorCheck:
    try:
        module = import_module(module_name)
    except Exception as error:
        return _DoctorCheck(label.lower(), "error", f"import failed: {error}")
    version = getattr(module, "__version__", None)
    detail = f"{version} imports successfully" if version else "imports successfully"
    return _DoctorCheck(label.lower(), "ok", detail)


def _distribution_check(distribution: str, label: str) -> _DoctorCheck:
    """Check an installed dependency without triggering its import-time setup."""
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return _DoctorCheck(label.lower(), "error", "distribution is not installed")
    except Exception as error:
        return _DoctorCheck(label.lower(), "error", f"metadata check failed: {error}")
    return _DoctorCheck(label.lower(), "ok", f"{version} is installed")


def _mediapipe_check() -> _DoctorCheck:
    """Import the exact Tasks API used by the primary backend, without noisy setup logs."""
    previous_logging_threshold = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stderr(StringIO()):
            tasks = import_module("mediapipe.tasks.python")
            vision = import_module("mediapipe.tasks.python.vision")
        required_symbols = (
            (tasks, "BaseOptions"),
            (vision, "FaceLandmarker"),
            (vision, "FaceLandmarkerOptions"),
            (vision, "RunningMode"),
        )
        missing = [name for module, name in required_symbols if not hasattr(module, name)]
        if missing:
            raise ImportError(f"required Tasks symbols are unavailable: {', '.join(missing)}")
        version = metadata.version("mediapipe")
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        release = tuple(int(part) for part in match.groups()) if match else None
        if release is None or not (release >= (0, 10, 9) and release < (1, 0, 0)):
            raise ImportError(
                f"version {version} is outside the supported {_MEDIAPIPE_RANGE} range"
            )
    except Exception as error:
        return _DoctorCheck("mediapipe", "error", f"Tasks import failed: {error}")
    finally:
        logging.disable(previous_logging_threshold)
    return _DoctorCheck("mediapipe", "ok", f"{version} Tasks API imports successfully")


def _cache_directory() -> Path:
    configured = os.environ.get("FRAME2FRAME_CACHE")
    return Path(configured) if configured else Path.home() / ".cache" / "frame2frame"


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _cache_check(cache: Path) -> _DoctorCheck:
    existing = _nearest_existing_parent(cache)
    if not existing.is_dir():
        return _DoctorCheck("cache", "error", f"parent is not a directory: {existing}")
    if not os.access(existing, os.W_OK | os.X_OK):
        return _DoctorCheck("cache", "error", f"not writable via {existing}")
    return _DoctorCheck("cache", "ok", f"writable via {existing} (target: {cache})")


def _model_check(cache: Path) -> _DoctorCheck:
    # Import constants from the owner so the doctor cannot drift from the
    # download contract used by the default backend.
    from .pose._facemesh import _MODEL_NAME, _MODEL_SHA256, _MODEL_SIZE

    model = cache.expanduser() / _MODEL_NAME
    if not model.exists():
        return _DoctorCheck(
            "mediapipe-model",
            "warning",
            f"not cached; first run will download {_MODEL_SIZE:,} verified bytes",
        )
    if not model.is_file():
        return _DoctorCheck("mediapipe-model", "error", f"cache path is not a file: {model}")
    try:
        valid = model.stat().st_size == _MODEL_SIZE and sha256_file(model) == _MODEL_SHA256
    except OSError as error:
        return _DoctorCheck("mediapipe-model", "error", f"cannot verify {model}: {error}")
    if not valid:
        return _DoctorCheck(
            "mediapipe-model",
            "warning",
            "cached file failed size or SHA-256 verification; the next run will replace it",
        )
    return _DoctorCheck("mediapipe-model", "ok", f"verified cached model: {model}")


def _ffmpeg_check() -> _DoctorCheck:
    executable = shutil.which("ffmpeg")
    if executable:
        return _DoctorCheck("ffmpeg", "ok", executable)
    return _DoctorCheck(
        "ffmpeg",
        "warning",
        "not found; required only for --preserve-audio and showcase generation",
    )


def _safely(name: str, check: Callable[[], _DoctorCheck]) -> _DoctorCheck:
    """Keep one broken probe from hiding the rest of the readiness report."""
    try:
        return check()
    except Exception as error:
        return _DoctorCheck(name, "error", f"check failed: {error}")


def collect_doctor_checks() -> tuple[_DoctorCheck, ...]:
    """Inspect the installed runtime without downloading models or opening devices."""
    cache = _cache_directory()
    dependencies = tuple(_dependency_check(module, label) for module, label in _REQUIRED_IMPORTS)
    distributions = tuple(
        _distribution_check(distribution, label) for distribution, label in _REQUIRED_DISTRIBUTIONS
    )
    return (
        _python_check(),
        *dependencies,
        _safely("mediapipe", _mediapipe_check),
        *distributions,
        _safely("cache", lambda: _cache_check(cache)),
        _safely("mediapipe-model", lambda: _model_check(cache)),
        _safely("ffmpeg", _ffmpeg_check),
    )


def format_doctor_report(checks: tuple[_DoctorCheck, ...]) -> str:
    lines = [f"frame2frame doctor {__version__}"]
    lines.extend(f"[{check.level}] {check.name}: {check.detail}" for check in checks)
    errors = sum(check.level == "error" for check in checks)
    warnings = sum(check.level == "warning" for check in checks)
    passed = len(checks) - errors - warnings
    lines.append(f"summary: {passed} ok, {warnings} warning, {errors} error")
    return "\n".join(lines)


def run_doctor() -> int:
    checks = collect_doctor_checks()
    print(format_doctor_report(checks))
    return 1 if any(check.level == "error" for check in checks) else 0
