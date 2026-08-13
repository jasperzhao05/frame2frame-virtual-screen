"""Internal dependency-free helpers for verified model downloads."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
import urllib.request
from pathlib import Path

from ._version import __version__

log = logging.getLogger("frame2frame")

_CHUNK_SIZE = 1024 * 1024


class DownloadIntegrityError(RuntimeError):
    """Raised when a downloaded artifact does not match its pinned digest."""


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(sha256: str, expected_size: int, timeout: float) -> str:
    if not isinstance(sha256, str):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    expected_digest = sha256.lower()
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be greater than zero")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise ValueError("expected_size must be a positive integer")
    return expected_digest


def _download_to(
    temporary: Path,
    request: urllib.request.Request,
    *,
    artifact_name: str,
    expected_size: int,
    timeout: float,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    downloaded = 0
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        temporary.open("wb") as stream,
    ):
        while chunk := response.read(_CHUNK_SIZE):
            downloaded += len(chunk)
            if downloaded > expected_size:
                raise DownloadIntegrityError(
                    f"download exceeded the expected size for {artifact_name}: "
                    f"{downloaded} > {expected_size} bytes"
                )
            stream.write(chunk)
            digest.update(chunk)
    return downloaded, digest.hexdigest()


def ensure_download(
    url: str,
    destination: str | Path,
    *,
    sha256: str,
    expected_size: int,
    timeout: float = 60.0,
) -> Path:
    """Return a verified cached artifact, downloading it atomically when needed.

    An existing cache entry is reused only when its size and digest match.
    Downloads are stopped if they exceed the pinned byte count, land in a
    unique sibling temporary file, and replace the destination only after the
    complete payload has been verified.
    """
    destination = Path(destination).expanduser()
    expected_digest = _validate_options(sha256, expected_size, timeout)
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and sha256_file(destination) == expected_digest
    ):
        return destination
    if destination.exists():
        log.warning(
            "cached artifact failed verification; downloading a fresh copy: %s",
            destination,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    request = urllib.request.Request(url, headers={"User-Agent": f"frame2frame/{__version__}"})
    try:
        log.info("downloading %s", destination.name)
        downloaded, actual_digest = _download_to(
            temporary,
            request,
            artifact_name=destination.name,
            expected_size=expected_size,
            timeout=timeout,
        )
        if downloaded != expected_size:
            raise DownloadIntegrityError(
                f"size mismatch for {destination.name}: expected {expected_size} bytes, "
                f"got {downloaded}"
            )
        if actual_digest != expected_digest:
            raise DownloadIntegrityError(
                f"SHA-256 mismatch for {destination.name}: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination
