import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from frame2frame import _media as media

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG or not FFPROBE,
    reason="ffmpeg and ffprobe are required for remux integration tests",
)


@pytest.fixture
def mux_files(tmp_path):
    files = SimpleNamespace(
        processed=tmp_path / "processed.mp4",
        source=tmp_path / "source.mp4",
        output=tmp_path / "output.mp4",
    )
    files.processed.write_bytes(b"video")
    files.source.write_bytes(b"source")
    files.output.write_bytes(b"known good")
    return files


def _run_ffmpeg(*arguments):
    assert FFMPEG
    subprocess.run(
        [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
    )


def _create_video(path, *, duration, color, audio_duration=None):
    command = [
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=32x24:r=10:d={duration}",
    ]
    if audio_duration is not None:
        command.extend(["-f", "lavfi", "-i", f"sine=frequency=1000:duration={audio_duration}"])
    command.extend(["-c:v", "mpeg4"])
    if audio_duration is None:
        command.append("-an")
    else:
        command.extend(["-c:a", "aac"])
    _run_ffmpeg(*command, str(path))


def _probe(path, *, entries, stream=None, count_frames=False, output_format=None):
    assert FFPROBE
    command = [FFPROBE, "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    if stream:
        command.extend(["-select_streams", stream])
    command.extend(
        [
            "-show_entries",
            entries,
            "-of",
            output_format or "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    return subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def test_staged_video_path_is_a_sibling_and_always_cleaned(tmp_path):
    output = tmp_path / "result.mp4"

    with media.staged_video_path(output) as staged:
        staged = Path(staged)
        assert staged.parent == tmp_path
        assert staged.suffix == ".mp4"
        assert staged.is_file()
        assert staged.stat().st_size == 0
        staged.write_bytes(b"partial")

    assert not staged.exists()


def test_install_video_atomically_replaces_existing_output(tmp_path):
    staged = tmp_path / ".stage.mp4"
    output = tmp_path / "output.mp4"
    staged.write_bytes(b"complete")
    output.write_bytes(b"old")

    result = media.install_video(staged, output)

    assert result == str(output)
    assert output.read_bytes() == b"complete"
    assert not staged.exists()


def test_install_video_rejects_empty_stage_without_touching_output(tmp_path):
    staged = tmp_path / ".stage.mp4"
    output = tmp_path / "output.mp4"
    staged.touch()
    output.write_bytes(b"known good")

    with pytest.raises(RuntimeError, match="missing or empty"):
        media.install_video(staged, output)

    assert output.read_bytes() == b"known good"


def test_mux_audio_atomically_replaces_output(monkeypatch, mux_files):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"video plus audio")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    result = media.mux_audio(
        mux_files.processed,
        mux_files.source,
        mux_files.output,
        duration_seconds=1.0,
        ffmpeg="/fake/ffmpeg",
    )

    assert result == str(mux_files.output)
    assert mux_files.output.read_bytes() == b"video plus audio"
    assert commands[0][1:-1] == [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mux_files.processed),
        "-i",
        str(mux_files.source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-t",
        "1.000000000",
    ]
    assert not list(mux_files.output.parent.glob(".output.mux-*"))


def test_mux_failure_preserves_existing_output(monkeypatch, mux_files):
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="bad stream"),
    )

    with pytest.raises(RuntimeError, match="bad stream"):
        media.mux_audio(
            mux_files.processed,
            mux_files.source,
            mux_files.output,
            duration_seconds=1.0,
            ffmpeg="/fake/ffmpeg",
        )

    assert mux_files.output.read_bytes() == b"known good"
    assert not list(mux_files.output.parent.glob(".output.mux-*"))


def test_mux_requires_ffmpeg_without_touching_output(monkeypatch, mux_files):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        media.mux_audio(
            mux_files.processed,
            mux_files.source,
            mux_files.output,
            duration_seconds=1.0,
        )

    assert mux_files.output.read_bytes() == b"known good"


@pytest.mark.integration
@requires_ffmpeg
@pytest.mark.parametrize(
    "source_has_audio",
    [
        pytest.param(False, id="silent-source"),
        pytest.param(True, id="source-with-audio"),
    ],
)
def test_remux_accepts_sources_with_or_without_audio(mux_files, source_has_audio):
    _create_video(mux_files.processed, duration=0.5, color="red")
    _create_video(
        mux_files.source,
        duration=0.5,
        color="black",
        audio_duration=0.5 if source_has_audio else None,
    )

    media.mux_audio(
        mux_files.processed,
        mux_files.source,
        mux_files.output,
        duration_seconds=0.5,
        ffmpeg=FFMPEG,
    )
    audio_stream = _probe(
        mux_files.output,
        stream="a",
        entries="stream=index",
        output_format="csv=p=0",
    )

    assert mux_files.output.stat().st_size > 0
    assert bool(audio_stream) is source_has_audio


@pytest.mark.integration
@requires_ffmpeg
@pytest.mark.parametrize(
    "audio_duration",
    [
        pytest.param(0.2, id="short-audio"),
        pytest.param(2.0, id="long-audio"),
    ],
)
def test_remux_duration_follows_processed_video_not_audio(mux_files, audio_duration):
    _create_video(mux_files.processed, duration=1, color="red")
    _create_video(
        mux_files.source,
        duration=1,
        color="black",
        audio_duration=audio_duration,
    )

    media.mux_audio(
        mux_files.processed,
        mux_files.source,
        mux_files.output,
        duration_seconds=1.0,
        ffmpeg=FFMPEG,
    )
    video_frames = _probe(
        mux_files.output,
        stream="v:0",
        entries="stream=nb_read_frames",
        count_frames=True,
    )
    output_duration = _probe(mux_files.output, entries="format=duration")

    assert int(video_frames) == 10
    assert float(output_duration) == pytest.approx(1.0, abs=0.12)


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param(None, id="missing"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_mux_rejects_invalid_duration_without_touching_output(mux_files, duration):
    with pytest.raises(ValueError, match="duration"):
        media.mux_audio(
            mux_files.processed,
            mux_files.source,
            mux_files.output,
            duration_seconds=duration,
            ffmpeg="/fake/ffmpeg",
        )

    assert mux_files.output.read_bytes() == b"known good"
