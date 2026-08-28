import numpy as np
import pytest

from frame2frame import pipeline
from frame2frame.config import ScreenConfig
from frame2frame.pose.base import FaceObservation, HeadPose
from tests.support.pipeline import (
    ClosingEstimator,
    EmptyReader,
)
from tests.support.pipeline import (
    minimal_pipeline_config as _minimal_config,
)


def test_input_output_collision_is_rejected_before_opening_reader(monkeypatch, tmp_path):
    source = tmp_path / "same.mp4"
    source.write_bytes(b"source")
    opened = False

    def open_reader(cfg):
        nonlocal opened
        opened = True
        raise AssertionError("must validate before opening")

    monkeypatch.setattr(pipeline, "_open_reader", open_reader)

    with pytest.raises(ValueError, match="different paths"):
        pipeline.run(_minimal_config(input=str(source), output=str(source)))

    assert not opened


def test_missing_ffmpeg_fails_before_opening_the_video(monkeypatch):
    opened = False

    def open_reader(cfg):
        nonlocal opened
        opened = True
        raise AssertionError("reader must not open before the ffmpeg preflight")

    monkeypatch.setattr(pipeline, "_open_reader", open_reader)
    monkeypatch.setattr(
        pipeline,
        "resolve_ffmpeg",
        lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg missing")),
    )

    with pytest.raises(RuntimeError, match="ffmpeg missing"):
        pipeline.run(_minimal_config(output="output.mp4", preserve_audio=True))

    assert not opened


def test_reader_is_released_when_estimator_creation_fails(monkeypatch):
    reader = EmptyReader()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(
        pipeline,
        "create_estimator",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("backend failed")),
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        pipeline.run(_minimal_config())

    assert reader.released


@pytest.mark.parametrize("backend", ["mediapipe", "mp", "facemesh"])
def test_owned_mediapipe_estimator_uses_the_renderer_focal_length(monkeypatch, backend):
    reader = EmptyReader()
    estimator = ClosingEstimator()
    created = {}
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)

    def create_estimator(name, **kwargs):
        created.update(name=name, kwargs=kwargs)
        return estimator

    monkeypatch.setattr(pipeline, "create_estimator", create_estimator)

    pipeline.run(
        _minimal_config(
            backend=backend,
            screen=ScreenConfig(focal_length=812.5),
        )
    )

    assert created == {
        "name": backend,
        "kwargs": {"fps": 30.0, "focal_length": 812.5},
    }


def test_renderer_focal_length_overrides_conflicting_backend_kwarg(monkeypatch):
    reader = EmptyReader()
    estimator = ClosingEstimator()
    created = {}
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)

    def create_estimator(name, **kwargs):
        created.update(kwargs)
        return estimator

    monkeypatch.setattr(pipeline, "create_estimator", create_estimator)

    pipeline.run(
        _minimal_config(
            backend_kwargs={"focal_length": 900.0},
            screen=ScreenConfig(focal_length=812.5),
        )
    )

    assert created["focal_length"] == 812.5


def test_reader_and_owned_estimator_close_when_texture_loading_fails(monkeypatch):
    reader = EmptyReader()
    estimator = ClosingEstimator()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_estimator", lambda *args, **kwargs: estimator)
    monkeypatch.setattr(
        pipeline,
        "_load_content",
        lambda cfg: (_ for _ in ()).throw(FileNotFoundError("texture")),
    )

    with pytest.raises(FileNotFoundError, match="texture"):
        pipeline.run(_minimal_config(draw_screen=True))

    assert reader.released
    assert estimator.closed


def test_configured_screen_video_is_pipeline_owned_and_closed(monkeypatch):
    reader = EmptyReader()
    estimator = ClosingEstimator()

    class ScreenVideo:
        def __init__(self, path, *, end_policy):
            self.path = path
            self.end_policy = end_policy
            self.closed = False

        def frame_at(self, request):
            return np.zeros((2, 2, 3), np.uint8)

        def close(self):
            self.closed = True

    created = []

    def open_screen_video(*args, **kwargs):
        source = ScreenVideo(*args, **kwargs)
        created.append(source)
        return source

    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "VideoContentSource", open_screen_video)

    pipeline.run(
        _minimal_config(
            draw_screen=True,
            screen=ScreenConfig(video_path="screen.mp4", video_end="loop"),
        ),
        estimator=estimator,
    )

    assert len(created) == 1
    assert created[0].path == "screen.mp4"
    assert created[0].end_policy == "loop"
    assert created[0].closed
    assert not estimator.closed


def test_injected_content_source_remains_caller_owned(monkeypatch):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    class InjectedContent:
        def __init__(self):
            self.closed = False

        def frame_at(self, request):
            return np.zeros((2, 2, 3), np.uint8)

        def close(self):
            self.closed = True

    source = InjectedContent()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: OneFrameReader())

    summary = pipeline.run(
        _minimal_config(draw_screen=True),
        estimator=ClosingEstimator(),
        content_source=source,
    )

    assert summary.content_samples == 1
    assert summary.render_samples == 0
    assert not source.closed


def test_latest_frame_source_rejects_file_processing_before_opening_reader(monkeypatch):
    opened = False

    def open_reader(cfg):
        nonlocal opened
        opened = True
        raise AssertionError("must reject before opening")

    monkeypatch.setattr(pipeline, "_open_reader", open_reader)

    with pytest.raises(ValueError, match="requires a webcam input"):
        pipeline.run(
            _minimal_config(draw_screen=True),
            content_source=pipeline.LatestFrameSource(),
        )

    assert not opened


def test_configured_screen_video_closes_when_sampling_fails(monkeypatch):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    class FailingScreenVideo:
        def __init__(self, path, *, end_policy):
            self.closed = False

        def frame_at(self, request):
            raise RuntimeError("content decode failed")

        def close(self):
            self.closed = True

    created = []

    def open_screen_video(*args, **kwargs):
        source = FailingScreenVideo(*args, **kwargs)
        created.append(source)
        return source

    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: OneFrameReader())
    monkeypatch.setattr(pipeline, "VideoContentSource", open_screen_video)

    with pytest.raises(RuntimeError, match="content decode failed"):
        pipeline.run(
            _minimal_config(
                draw_screen=True,
                screen=ScreenConfig(video_path="screen.mp4"),
            ),
            estimator=ClosingEstimator(),
        )

    assert len(created) == 1
    assert created[0].closed


def test_injected_and_configured_content_conflict_before_opening_reader(monkeypatch):
    opened = False

    def open_reader(cfg):
        nonlocal opened
        opened = True
        raise AssertionError("must reject before opening")

    monkeypatch.setattr(pipeline, "_open_reader", open_reader)

    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.run(
            _minimal_config(
                draw_screen=True,
                screen=ScreenConfig(texture_path="screen.png"),
            ),
            content_source=lambda request: np.zeros((2, 2, 3), np.uint8),
        )

    assert not opened


def test_resources_close_when_writer_initialization_fails(monkeypatch, tmp_path):
    reader = EmptyReader()
    estimator = ClosingEstimator()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_estimator", lambda *args, **kwargs: estimator)
    monkeypatch.setattr(
        pipeline,
        "VideoWriter",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer failed")),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        pipeline.run(_minimal_config(output=str(tmp_path / "output.mp4")))

    assert reader.released
    assert estimator.closed


def test_inference_mean_is_accumulated_without_storing_frame_timings(monkeypatch):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    reader = OneFrameReader()
    estimator = ClosingEstimator()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)

    summary = pipeline.run(_minimal_config(), estimator=estimator)

    assert summary.frames == 1
    assert summary.mean_inference_ms >= 0
    assert reader.released
    assert not estimator.closed  # injected estimators remain caller-owned


def test_display_stop_closes_preview_and_stops_decoding(monkeypatch):
    class PreviewReader(EmptyReader):
        def __iter__(self):
            for index in range(5):
                yield np.full((12, 20, 3), index, np.uint8)

    reader = PreviewReader()
    estimator = ClosingEstimator()
    shown = []
    preview_closed = False

    def show(frame):
        shown.append(int(frame[0, 0, 0]))
        return len(shown) == 2

    def close_preview():
        nonlocal preview_closed
        preview_closed = True

    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "_show", show)
    monkeypatch.setattr(pipeline, "_destroy_windows", close_preview)

    summary = pipeline.run(_minimal_config(display=True), estimator=estimator)

    assert shown == [0, 1]
    assert summary.frames == 2
    assert reader.released
    assert preview_closed
    assert not estimator.closed


@pytest.mark.parametrize(
    "observation",
    [
        pytest.param(
            FaceObservation(
                HeadPose(float("nan"), 0, 0),
                (10, 6),
                2,
                (8, 4, 12, 8),
            ),
            id="nonfinite-yaw",
        ),
        pytest.param(
            FaceObservation(HeadPose(0, 0, 0), (10, 6), 0, (8, 4, 12, 8)),
            id="nonpositive-face-size",
        ),
        pytest.param(
            FaceObservation(HeadPose(0, 0, 0), (10, 6), 2, (12, 4, 8, 8)),
            id="inverted-bbox",
        ),
    ],
)
def test_invalid_backend_observation_is_rejected(monkeypatch, observation):
    class OneFrameReader(EmptyReader):
        def __iter__(self):
            yield np.zeros((12, 20, 3), np.uint8)

    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: OneFrameReader())

    class InvalidEstimator:
        def estimate(self, frame):
            return observation

    with pytest.raises(ValueError, match="pose backend"):
        pipeline.run(_minimal_config(), estimator=InvalidEstimator())


def test_cleanup_failure_is_logged_without_skipping_other_cleanup(monkeypatch, caplog):
    class BrokenReader(EmptyReader):
        def release(self):
            self.released = True
            raise RuntimeError("reader close failed")

    reader = BrokenReader()
    estimator = ClosingEstimator()
    monkeypatch.setattr(pipeline, "_open_reader", lambda cfg: reader)
    monkeypatch.setattr(pipeline, "create_estimator", lambda *args, **kwargs: estimator)

    summary = pipeline.run(_minimal_config())

    assert summary.frames == 0
    assert reader.released
    assert estimator.closed
    assert "failed to close video reader" in caplog.text
