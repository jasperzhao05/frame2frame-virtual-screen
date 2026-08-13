from types import SimpleNamespace

import numpy as np

from frame2frame import _diagnostics as diagnostics
from frame2frame import pipeline
from frame2frame._diagnostics import _AngleDiagnostics, _break_frame_gaps
from tests.support.pipeline import (
    RecordingWriter,
    SequenceReader,
)
from tests.support.pipeline import (
    make_observation as _observation,
)
from tests.support.pipeline import (
    timing_pipeline_config as _config,
)


def test_plot_diagnostics_are_bounded_and_delay_aligned(monkeypatch, tmp_path):
    reader = SequenceReader(12)
    captured = {}
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline,
        "create_filter",
        lambda fps, cfg: SimpleNamespace(
            group_delay=0,
            update=lambda yaw, pitch, roll, *, dt=None: (yaw, pitch, roll),
            update_position=lambda cx, cy, size, *, dt=None: (cx, cy, size),
            reset=lambda: None,
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "plot_angles",
        lambda raw, smoothed, path, *, frame_indices: captured.update(
            frames=frame_indices,
            raw=raw,
            smoothed=smoothed,
            path=path,
        ),
    )

    class Estimator:
        def estimate(self, frame):
            return _observation(int(frame[0, 0, 0]))

    plot_path = tmp_path / "diagnostics.png"
    pipeline.run(
        _config(
            output=str(tmp_path / "output.mp4"),
            draw_axis=False,
            plot_path=str(plot_path),
            max_plot_samples=3,
        ),
        estimator=Estimator(),
    )

    assert [row[0] for row in captured["raw"]] == [90.0, 100.0, 110.0]
    assert [row[0] for row in captured["smoothed"]] == [90.0, 100.0, 110.0]
    assert captured["frames"] == [9, 10, 11]
    assert captured["path"] == str(plot_path)


def test_diagnostics_mark_missing_source_frames_without_compressing_time():
    diagnostics = _AngleDiagnostics(enabled=True, max_samples=10)

    diagnostics.record(4, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 0, 0, 1))
    diagnostics.record(5, None, (1.0, 2.0, 3.0, 0, 0, 1))
    diagnostics.record(6, (4.0, 5.0, 6.0), (4.0, 5.0, 6.0, 0, 0, 1))

    assert list(diagnostics.frames) == [4, 5, 6]
    assert np.isnan(diagnostics.raw[1]).all()


def test_plot_series_breaks_at_unobserved_frame_ranges():
    frames, values = _break_frame_gaps(
        [4, 5, 20],
        [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (5.0, 6.0, 7.0)],
    )

    assert frames == [4, 5, 12.5, 20]
    assert np.isnan(values[2]).all()


def test_diagnostics_break_adjacent_indices_when_tracking_segment_resets():
    diagnostics = _AngleDiagnostics(enabled=True, max_samples=10)
    values = (1.0, 2.0, 3.0, 0, 0, 1)
    diagnostics.record(4, values[:3], values)
    diagnostics.end_segment()
    diagnostics.record(5, values[:3], values)

    assert list(diagnostics.frames) == [4, 4.5, 5]
    assert np.isnan(diagnostics.raw[1]).all()
