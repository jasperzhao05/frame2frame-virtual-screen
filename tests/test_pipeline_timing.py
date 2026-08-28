import numpy as np
import pytest

from frame2frame import pipeline
from frame2frame._diagnostics import _AngleDiagnostics
from frame2frame._tracking import _TemporalTracker
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


class DelayedFilter:
    group_delay = 2

    def __init__(self):
        self.history = []
        self.total_updates = 0
        self.reset_count = 0

    def update(self, yaw, pitch, roll, *, dt=None):
        del dt
        self.total_updates += 1
        self.history.append((yaw, pitch, roll))
        delayed_index = max(0, len(self.history) - 1 - self.group_delay)
        return self.history[delayed_index]

    def update_position(self, cx, cy, size, *, dt=None):
        del dt
        return cx, cy, size

    def reset(self):
        self.history.clear()
        self.reset_count += 1


def test_delay_compensation_advances_on_every_video_frame(monkeypatch, tmp_path):
    reader = SequenceReader(5)
    smoother = DelayedFilter()
    rendered = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_filter", lambda fps, cfg: smoother)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline.render,
        "draw_pose_axis",
        lambda frame, yaw, *args: rendered.append((int(frame[0, 0, 0]), yaw)),
    )

    class Estimator:
        def estimate(self, frame):
            value = int(frame[0, 0, 0])
            return None if value == 2 else _observation(value)

    summary = pipeline.run(
        _config(output=str(tmp_path / "output.mp4")),
        estimator=Estimator(),
    )

    assert rendered == [(0, 0.0), (1, 10.0), (3, 30.0), (4, 40.0)]
    assert RecordingWriter.instances[0].frames == [0, 1, 2, 3, 4]
    assert summary.frames == 5
    assert summary.faces == 4
    assert reader.released
    assert len(smoother.history) == 7  # five frames plus two EOF padding samples


def test_dynamic_content_uses_the_pending_packet_media_time_with_fir(
    monkeypatch,
    tmp_path,
):
    reader = SequenceReader(6)
    smoother = DelayedFilter()
    requests = []
    rendered = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_filter", lambda fps, cfg: smoother)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline.render,
        "draw_virtual_screen",
        lambda frame, observation, cfg, content, *, opacity: rendered.append(
            (int(frame[0, 0, 0]), int(content.bgr[0, 0, 0]))
        ),
    )

    def content_source(request):
        requests.append((request.frame_index, request.media_time_seconds, request.live))
        return np.full((2, 2, 3), request.frame_index, np.uint8)

    class Estimator:
        def estimate(self, frame):
            return _observation(int(frame[0, 0, 0]))

    summary = pipeline.run(
        _config(
            output=str(tmp_path / "output.mp4"),
            draw_axis=False,
            draw_screen=True,
        ),
        estimator=Estimator(),
        content_source=content_source,
    )

    assert rendered == [(index, index) for index in range(6)]
    assert requests == [(index, pytest.approx(index / reader.fps), False) for index in range(6)]
    assert summary.content_samples == 6
    assert summary.render_samples == 6
    assert summary.mean_content_ms >= 0
    assert summary.mean_render_ms >= 0


def test_dynamic_content_clock_advances_through_detection_gaps(monkeypatch, tmp_path):
    reader = SequenceReader(8)
    requested = []
    rendered = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline.render,
        "draw_virtual_screen",
        lambda frame, observation, cfg, content, *, opacity: rendered.append(
            (int(frame[0, 0, 0]), int(content.bgr[0, 0, 0]))
        ),
    )

    def content_source(request):
        requested.append(request.frame_index)
        return np.full((2, 2, 3), request.frame_index, np.uint8)

    class Estimator:
        def estimate(self, frame):
            index = int(frame[0, 0, 0])
            return _observation(index) if index in {0, 7} else None

    pipeline.run(
        _config(
            output=str(tmp_path / "output.mp4"),
            draw_axis=False,
            draw_screen=True,
            dropout_hold_seconds=0.2,
            dropout_reset_seconds=0.5,
        ),
        estimator=Estimator(),
        content_source=content_source,
    )

    assert requested == list(range(8))
    assert all(scene_index == content_index for scene_index, content_index in rendered)
    assert rendered[-1] == (7, 7)


def test_content_sampler_reprepares_a_reused_grayscale_buffer():
    buffer = np.zeros((2, 3), np.uint8)
    sampler = pipeline._ContentSampler(lambda request: buffer)

    first = sampler.sample(pipeline.ContentRequest(0, 0.0, False))
    buffer.fill(91)
    second = sampler.sample(pipeline.ContentRequest(1, 0.1, False))

    assert first is not None and second is not None
    assert int(first.bgr[0, 0, 0]) == 0
    assert int(second.bgr[0, 0, 0]) == 91


def test_content_sampler_reprepares_a_reused_bgra_buffer():
    buffer = np.zeros((2, 3, 4), np.uint8)
    sampler = pipeline._ContentSampler(lambda request: buffer)

    first = sampler.sample(pipeline.ContentRequest(0, 0.0, False))
    buffer[..., :3] = (11, 22, 33)
    buffer[..., 3] = 204
    second = sampler.sample(pipeline.ContentRequest(1, 0.1, False))

    assert first is not None and first.alpha is not None
    assert second is not None and second.alpha is not None
    np.testing.assert_array_equal(first.bgr[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(second.bgr[0, 0], (11, 22, 33))
    assert float(first.alpha[0, 0]) == 0.0
    assert float(second.alpha[0, 0]) == pytest.approx(0.8)


def test_sparse_detections_keep_wall_clock_delay_and_queue_bounded(monkeypatch, tmp_path):
    reader = SequenceReader(75)
    reader.fps = 30.0
    smoother = DelayedFilter()
    written = []
    observed_lag = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_filter", lambda fps, cfg: smoother)

    class Writer(RecordingWriter):
        def write(self, frame):
            super().write(frame)
            written.append(int(frame[0, 0, 0]))

    monkeypatch.setattr(pipeline, "VideoWriter", Writer)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(pipeline.render, "draw_pose_axis", lambda *args: None)

    class Estimator:
        def estimate(self, frame):
            index = int(frame[0, 0, 0])
            # This includes the frame currently being inferred. With a
            # two-frame group delay, at most three decoded frames can be
            # outstanding even when detections arrive only every 15 frames.
            observed_lag.append(index + 1 - len(written))
            return _observation(index) if index % 15 == 0 else None

    pipeline.run(
        _config(output=str(tmp_path / "output.mp4")),
        estimator=Estimator(),
    )

    assert max(observed_lag) <= smoother.group_delay + 1
    assert written == list(range(75))
    assert smoother.total_updates == 75 + smoother.group_delay * (smoother.reset_count + 1)


def test_long_dropout_flushes_pending_segment_and_resets_filter(monkeypatch, tmp_path):
    reader = SequenceReader(7)
    reader.fps = 4.0  # reset after two consecutive misses
    smoother = DelayedFilter()
    written = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_filter", lambda fps, cfg: smoother)

    class Writer(RecordingWriter):
        def write(self, frame):
            super().write(frame)
            written.append(int(frame[0, 0, 0]))

    monkeypatch.setattr(pipeline, "VideoWriter", Writer)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(pipeline.render, "draw_pose_axis", lambda *args: None)

    class Estimator:
        def estimate(self, frame):
            value = int(frame[0, 0, 0])
            if value == 4:
                # Frames 0-3 must have been released as soon as the miss
                # threshold was reached, before decoding continues.
                assert written == [0, 1, 2, 3]
            return None if value in {2, 3, 4} else _observation(value)

    pipeline.run(
        _config(output=str(tmp_path / "output.mp4")),
        estimator=Estimator(),
    )

    assert written == list(range(7))
    assert smoother.reset_count == 1


def test_short_detection_gap_holds_and_fades_only_the_screen(monkeypatch, tmp_path):
    reader = SequenceReader(7)
    screens = []
    axes = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline,
        "_load_content",
        lambda cfg: np.zeros((1, 1, 3), np.uint8),
    )
    monkeypatch.setattr(
        pipeline.render,
        "draw_virtual_screen",
        lambda frame, observation, cfg, content, *, opacity: screens.append(
            (int(frame[0, 0, 0]), observation.pose.yaw, opacity)
        ),
    )
    monkeypatch.setattr(
        pipeline.render,
        "draw_pose_axis",
        lambda frame, yaw, *args: axes.append((int(frame[0, 0, 0]), yaw)),
    )

    class Estimator:
        def estimate(self, frame):
            index = int(frame[0, 0, 0])
            return _observation(index) if index in {0, 6} else None

    summary = pipeline.run(
        _config(
            output=str(tmp_path / "output.mp4"),
            draw_screen=True,
            dropout_hold_seconds=0.2,
            dropout_reset_seconds=0.5,
        ),
        estimator=Estimator(),
    )

    assert [row[0] for row in screens] == [0, 1, 2, 3, 4, 6]
    assert [row[2] for row in screens[:3]] == [1.0, 1.0, 1.0]
    assert screens[3][2] == pytest.approx(2 / 3)
    assert screens[4][2] == pytest.approx(1 / 3)
    assert axes == [(0, 0.0), (6, 60.0)]
    assert summary.faces == 2


def test_file_source_passes_deterministic_frame_timestamps(monkeypatch, tmp_path):
    reader = SequenceReader(3)
    timestamps = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))

    class Estimator:
        def estimate_at(self, frame, timestamp_ms):
            timestamps.append(timestamp_ms)
            return None

    pipeline.run(
        _config(output=str(tmp_path / "output.mp4")),
        estimator=Estimator(),
    )

    assert timestamps == pytest.approx([100.0, 200.0, 300.0])


def test_webcam_source_passes_monotonic_capture_timestamps(monkeypatch, tmp_path):
    reader = SequenceReader(2)
    timestamps = []
    content_requests = []
    clock = iter([10.0, 10.05, 10.17])
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(clock))

    class Estimator:
        def estimate_at(self, frame, timestamp_ms):
            timestamps.append(timestamp_ms)
            return None

    pipeline.run(
        _config(
            input=None,
            webcam=0,
            output=str(tmp_path / "output.mp4"),
            draw_screen=True,
        ),
        estimator=Estimator(),
        content_source=lambda request: content_requests.append(request) or None,
    )

    assert timestamps == pytest.approx([50.0, 170.0])
    assert [request.frame_index for request in content_requests] == [0, 1]
    assert [request.media_time_seconds for request in content_requests] == pytest.approx(
        [0.05, 0.17]
    )
    assert all(request.live for request in content_requests)


def test_webcam_dropout_fade_uses_elapsed_time_not_nominal_frame_count(
    monkeypatch,
    tmp_path,
):
    reader = SequenceReader(3)
    clock = iter([10.0, 10.0, 10.1, 10.45])
    screens = []
    RecordingWriter.instances.clear()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoWriter", RecordingWriter)
    monkeypatch.setattr(pipeline, "install_video", lambda processed, output: str(output))
    monkeypatch.setattr(
        pipeline,
        "_load_content",
        lambda cfg: np.zeros((1, 1, 3), np.uint8),
    )
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        pipeline.render,
        "draw_virtual_screen",
        lambda frame, observation, cfg, content, *, opacity: screens.append(opacity),
    )

    class Estimator:
        def estimate(self, frame):
            return _observation(0) if int(frame[0, 0, 0]) == 0 else None

    pipeline.run(
        _config(
            input=None,
            webcam=0,
            output=str(tmp_path / "output.mp4"),
            draw_screen=True,
            dropout_hold_seconds=0.2,
            dropout_reset_seconds=0.5,
        ),
        estimator=Estimator(),
    )

    assert screens == pytest.approx([1.0, 1.0, 1 / 6])


def test_reacquisition_after_wall_clock_reset_starts_a_fresh_filter_segment():
    smoother = DelayedFilter()
    cfg = _config(
        input=None,
        webcam=0,
        dropout_hold_seconds=0.2,
        dropout_reset_seconds=0.5,
    )
    tracking = _TemporalTracker(
        cfg,
        fps=10.0,
        smoother=smoother,
        diagnostics=_AngleDiagnostics(enabled=False, max_samples=10),
    )

    tracking.push(np.zeros((2, 2, 3), np.uint8), _observation(0), 0, 0.0)
    tracking.push(np.zeros((2, 2, 3), np.uint8), None, 1, 0.3)
    ready = tracking.push(
        np.zeros((2, 2, 3), np.uint8),
        _observation(10),
        2,
        0.6,
    )

    assert smoother.reset_count == 1
    assert ready[-1].detected is True
    assert ready[-1].observation.pose.yaw == 100.0


def test_nominal_timestamp_fallback_counts_the_reacquisition_frame():
    smoother = DelayedFilter()
    cfg = _config(
        input=None,
        webcam=0,
        dropout_hold_seconds=0.2,
        dropout_reset_seconds=0.5,
    )
    tracking = _TemporalTracker(
        cfg,
        fps=10.0,
        smoother=smoother,
        diagnostics=_AngleDiagnostics(enabled=False, max_samples=10),
    )
    frame = np.zeros((2, 2, 3), np.uint8)

    tracking.push(frame.copy(), _observation(0))
    for _ in range(4):
        tracking.push(frame.copy(), None)
    ready = tracking.push(frame.copy(), _observation(10))

    assert smoother.reset_count == 1
    assert ready[-1].observation.pose.yaw == 100.0


def test_invalid_timestamp_does_not_partially_replace_tracking_state():
    smoother = DelayedFilter()
    cfg = _config(input=None, webcam=0)
    tracking = _TemporalTracker(
        cfg,
        fps=10.0,
        smoother=smoother,
        diagnostics=_AngleDiagnostics(enabled=False, max_samples=10),
    )
    frame = np.zeros((2, 2, 3), np.uint8)
    tracking.push(frame.copy(), _observation(1), timestamp_seconds=1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        tracking.push(frame.copy(), _observation(9), timestamp_seconds=1.0)

    assert tracking.anchor is not None
    assert tracking.anchor.sample.yaw == 10.0
    assert tracking.last_timestamp_seconds == 1.0
