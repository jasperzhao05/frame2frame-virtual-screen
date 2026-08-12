from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from frame2frame import _diagnostics as diagnostics
from frame2frame import pipeline
from frame2frame._diagnostics import plot_angles
from frame2frame.pose.base import FaceObservation, HeadPose
from tests.support.pipeline import (
    ClosingEstimator,
    EmptyReader,
)
from tests.support.pipeline import (
    minimal_pipeline_config as _minimal_config,
)


def test_processing_failure_does_not_replace_existing_output(monkeypatch, tmp_path):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    class StagingWriter:
        def __init__(self, path, fps, size):
            self.path = str(path)

        def write(self, frame):
            pass

        def release(self):
            # Simulate an encoder that managed to flush a partial file.
            Path(self.path).write_bytes(b"partial")

    reader = OneFrameReader()
    output = tmp_path / "output.mp4"
    output.write_bytes(b"known good")
    estimator = ClosingEstimator(error=RuntimeError("inference failed"))
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", StagingWriter)

    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.run(
            _minimal_config(output=str(output)),
            estimator=estimator,
        )

    assert output.read_bytes() == b"known good"
    assert reader.released


def test_plot_failure_does_not_replace_existing_output(monkeypatch, tmp_path):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    class StagingWriter:
        def __init__(self, path, fps, size):
            self.path = str(path)
            self.released = False

        def write(self, frame):
            pass

        def release(self):
            if not self.released:
                Path(self.path).write_bytes(b"complete video")
                self.released = True

    class FaceEstimator:
        def estimate(self, frame):
            return FaceObservation(HeadPose(0, 0, 0), (10, 6), 2, (8, 4, 12, 8))

    reader = OneFrameReader()
    output = tmp_path / "output.mp4"
    output.write_bytes(b"known good")
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", StagingWriter)
    monkeypatch.setattr(
        diagnostics,
        "plot_angles",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        pipeline.run(
            _minimal_config(output=str(output), plot_path=str(tmp_path / "plot.png")),
            estimator=FaceEstimator(),
        )

    assert output.read_bytes() == b"known good"


def test_plot_is_installed_atomically(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    plot = tmp_path / "angles.png"
    plot.write_bytes(b"known good plot")
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save failed")),
    )

    with pytest.raises(RuntimeError, match="save failed"):
        plot_angles([(1.0, 2.0, 3.0)], [(1.0, 2.0, 3.0)], plot)

    assert plot.read_bytes() == b"known good plot"
    assert not list(tmp_path.glob(".angles.plot-*"))


def test_plot_initialization_failure_cleans_temporary_file(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    plot = tmp_path / "angles.png"
    plot.write_bytes(b"known good plot")
    monkeypatch.setattr(
        plt,
        "subplots",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        plot_angles([(1.0, 2.0, 3.0)], [(1.0, 2.0, 3.0)], plot)

    assert plot.read_bytes() == b"known good plot"
    assert not list(tmp_path.glob(".angles.plot-*"))


def test_empty_diagnostics_replace_a_stale_plot(tmp_path):
    plot = tmp_path / "angles.png"
    plot.write_bytes(b"stale plot")

    plot_angles([], [], plot)

    assert plot.read_bytes().startswith(b"\x89PNG")


def test_audio_remux_occurs_only_after_writer_release(monkeypatch, tmp_path):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    reader = OneFrameReader()
    estimator = ClosingEstimator()
    final = tmp_path / "output.mp4"
    events = []

    class Writer:
        def __init__(self, path, fps, size):
            self.path = path
            self.released = False

        def write(self, frame):
            pass

        def release(self):
            if not self.released:
                events.append("release")
                self.released = True

    @contextmanager
    def staged(path):
        yield str(tmp_path / "video-stage.mp4")

    def remux(processed, source, output, *, duration_seconds, ffmpeg):
        assert events == ["release"]
        assert duration_seconds == pytest.approx(1 / 30)
        assert ffmpeg == "/test/ffmpeg"
        events.append("remux")
        return str(output)

    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_estimator", lambda *args, **kwargs: estimator)
    monkeypatch.setattr(pipeline, "VideoWriter", Writer)
    monkeypatch.setattr(pipeline, "staged_video_path", staged)
    monkeypatch.setattr(pipeline, "mux_audio", remux)
    monkeypatch.setattr(pipeline, "resolve_ffmpeg", lambda: "/test/ffmpeg")

    summary = pipeline.run(_minimal_config(output=str(final), preserve_audio=True))

    assert events == ["release", "remux"]
    assert summary.output == str(final)
    assert summary.audio_remuxed is True
    assert reader.released
    assert estimator.closed
