import hashlib
import io

import pytest

from frame2frame import __version__
from frame2frame._downloads import DownloadIntegrityError, ensure_download, sha256_file

_URL = "https://example.test/model"


def _response(monkeypatch, payload):
    monkeypatch.setattr(
        "frame2frame._downloads.urllib.request.urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )


def _download(destination, *, digest="0" * 64, size=1, timeout=60.0):
    return ensure_download(
        _URL,
        destination,
        sha256=digest,
        expected_size=size,
        timeout=timeout,
    )


def test_verified_download_is_atomic_and_reused(tmp_path, monkeypatch):
    payload = b"verified model artifact"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "model.bin"
    _response(monkeypatch, payload)

    result = _download(destination, digest=expected, size=len(payload))
    assert result == destination
    assert destination.read_bytes() == payload
    assert sha256_file(destination) == expected

    monkeypatch.setattr(
        "frame2frame._downloads.urllib.request.urlopen",
        lambda request, timeout: pytest.fail("verified cache should not be downloaded again"),
    )
    result = _download(destination, digest=expected, size=len(payload))
    assert result == destination


def test_download_user_agent_tracks_runtime_version(tmp_path, monkeypatch):
    payload = b"model"
    captured = {}

    def response(request, timeout):
        captured["user_agent"] = request.get_header("User-agent")
        return io.BytesIO(payload)

    monkeypatch.setattr("frame2frame._downloads.urllib.request.urlopen", response)
    _download(
        tmp_path / "model.bin",
        digest=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )

    assert captured["user_agent"] == f"frame2frame/{__version__}"


def test_digest_mismatch_does_not_replace_existing_cache(tmp_path, monkeypatch):
    destination = tmp_path / "model.bin"
    destination.write_bytes(b"old cache")
    payload = b"wrong new payload"
    _response(monkeypatch, payload)

    with pytest.raises(DownloadIntegrityError, match="SHA-256 mismatch"):
        _download(
            destination,
            digest=hashlib.sha256(b"expected payload").hexdigest(),
            size=len(payload),
        )

    assert destination.read_bytes() == b"old cache"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("0" * 63, id="wrong-length"),
        pytest.param("g" * 64, id="non-hex"),
    ],
)
def test_invalid_digest_is_rejected_before_network_access(tmp_path, monkeypatch, digest):
    monkeypatch.setattr(
        "frame2frame._downloads.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("invalid input must not reach the network"),
    )

    with pytest.raises(ValueError, match="sha256"):
        _download(tmp_path / "model.bin", digest=digest)


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(True, id="boolean"),
    ],
)
def test_invalid_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        _download(tmp_path / "model.bin", timeout=timeout)


@pytest.mark.parametrize(
    "expected_size",
    [
        pytest.param(None, id="missing"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="noninteger"),
        pytest.param(True, id="boolean"),
    ],
)
def test_invalid_expected_size_is_rejected(tmp_path, expected_size):
    with pytest.raises(ValueError, match="expected_size"):
        _download(tmp_path / "model.bin", size=expected_size)


def test_oversized_response_is_stopped_and_existing_cache_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / "model.bin"
    destination.write_bytes(b"known good cache")
    payload = b"x" * 32
    _response(monkeypatch, payload)

    with pytest.raises(DownloadIntegrityError, match="exceeded"):
        _download(
            destination,
            digest=hashlib.sha256(b"expected").hexdigest(),
            size=8,
        )

    assert destination.read_bytes() == b"known good cache"
    assert not list(tmp_path.glob(".*.tmp"))


def test_interrupted_download_cleans_temporary_file(tmp_path, monkeypatch):
    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "frame2frame._downloads.urllib.request.urlopen",
        lambda request, timeout: InterruptedResponse(),
    )

    with pytest.raises(KeyboardInterrupt):
        _download(tmp_path / "model.bin", size=8)

    assert not list(tmp_path.glob(".*.tmp"))
