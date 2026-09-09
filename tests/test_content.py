import threading
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from frame2frame import content


def _request(index: int, timestamp: float, *, live: bool = False) -> content.ContentRequest:
    return content.ContentRequest(index, timestamp, live)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_index": -1}, "frame_index"),
        ({"frame_index": True}, "frame_index"),
        ({"media_time_seconds": -0.1}, "media_time_seconds"),
        ({"media_time_seconds": float("nan")}, "media_time_seconds"),
        ({"media_time_seconds": float("inf")}, "media_time_seconds"),
        ({"live": 1}, "live"),
    ],
)
def test_content_request_rejects_invalid_fields(kwargs, message):
    values = {"frame_index": 0, "media_time_seconds": 0.0, "live": False}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        content.ContentRequest(**values)


def test_content_request_is_frozen_and_normalizes_time():
    request = content.ContentRequest(3, 2, True)

    assert request.media_time_seconds == 2.0
    with pytest.raises(FrozenInstanceError):
        request.frame_index = 4


def test_latest_source_is_empty_until_a_frame_is_published():
    source = content.LatestFrameSource()

    assert isinstance(source, content.ContentSource)
    assert source.frame_at(_request(0, 0.0, live=True)) is None


def test_latest_source_safe_copy_isolated_from_producer_mutation():
    source = content.LatestFrameSource()
    original = np.full((2, 3, 3), 7, np.uint8)

    source.publish(original)
    original.fill(99)
    snapshot = source.frame_at(_request(0, 0.0, live=True))

    assert snapshot is not original
    np.testing.assert_array_equal(snapshot, np.full((2, 3, 3), 7, np.uint8))


def test_latest_source_explicit_zero_copy_and_latest_wins():
    source = content.LatestFrameSource()
    first = np.full((2, 3, 3), 1, np.uint8)
    second = np.full((2, 3, 3), 2, np.uint8)

    source.publish(first, copy=False)
    assert source.frame_at(_request(0, 0.0, live=True)) is first
    source.publish(second, copy=False)

    assert source.frame_at(_request(1, 0.1, live=True)) is second
    source.clear()
    assert source.frame_at(_request(2, 0.2, live=True)) is None


def test_latest_source_publishes_complete_snapshots_across_threads():
    source = content.LatestFrameSource()
    published = threading.Barrier(2)
    sampled = threading.Barrier(2)

    def produce():
        for value in range(20):
            source.publish(np.full((12, 16, 3), value, np.uint8))
            published.wait()
            sampled.wait()

    producer = threading.Thread(target=produce)
    producer.start()
    complete = []
    for value in range(20):
        published.wait()
        snapshot = source.frame_at(_request(value, value / 30, live=True))
        complete.append(snapshot is not None and bool(np.all(snapshot == value)))
        sampled.wait()
    producer.join(timeout=2)

    assert not producer.is_alive()
    assert all(complete)


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(np.empty((0, 3, 3), np.uint8), id="empty"),
        pytest.param(np.zeros((2,), np.uint8), id="one-dimensional"),
        pytest.param(np.zeros((2, 3, 2), np.uint8), id="two-channel"),
    ],
)
def test_latest_source_rejects_invalid_frames(frame):
    with pytest.raises(ValueError, match="content frame"):
        content.LatestFrameSource().publish(frame)


def test_latest_source_requires_numpy_and_boolean_copy_flag():
    source = content.LatestFrameSource()

    with pytest.raises(TypeError, match="NumPy"):
        source.publish([[1, 2]])
    with pytest.raises(ValueError, match="copy"):
        source.publish(np.zeros((2, 2, 3), np.uint8), copy=1)


class _FakeReader:
    fps = 2.0
    frame_count = 3

    def __init__(self, path, *, frames=None, fps=None):
        self.path = str(path)
        self.frames = list(_video_frames() if frames is None else frames)
        if fps is not None:
            self.fps = fps
        self.released = False

    def __iter__(self):
        return iter(self.frames)

    def release(self):
        self.released = True


def _video_frames():
    return [np.full((2, 3, 3), value, np.uint8) for value in range(3)]


def _reader_factory(monkeypatch, *, frames=None, fps=2.0):
    instances = []

    def open_reader(path):
        reader = _FakeReader(path, frames=frames, fps=fps)
        instances.append(reader)
        return reader

    monkeypatch.setattr(content, "VideoReader", open_reader)
    return instances


def _value(frame):
    return None if frame is None else int(frame[0, 0, 0])


def test_video_source_uses_floor_sampling_and_reuses_lower_rate_frames(monkeypatch):
    instances = _reader_factory(monkeypatch)
    source = content.VideoContentSource("content.mp4")
    times = [0.0, 0.49, 0.5, 0.99, 1.0]

    actual = [_value(source.frame_at(_request(index, time))) for index, time in enumerate(times)]

    assert actual == [0, 0, 1, 1, 2]
    assert len(instances) == 1
    source.close()
    assert instances[0].released


def test_video_source_treats_exact_decimal_boundary_as_next_frame(monkeypatch):
    frames = [np.full((2, 3, 3), value, np.uint8) for value in range(6)]
    _reader_factory(monkeypatch, frames=frames, fps=10.0)
    source = content.VideoContentSource("content.mp4")

    assert _value(source.frame_at(_request(0, 0.3))) == 3

    source.close()


@pytest.mark.parametrize(
    ("scene_fps", "content_fps", "expected"),
    [
        pytest.param(30.0, 24.0, [0, 0, 1, 2, 3, 4], id="content-slower"),
        pytest.param(24.0, 30.0, [0, 1, 2, 3, 5, 6], id="content-faster"),
    ],
)
def test_video_source_samples_different_frame_rates_without_future_frames(
    monkeypatch,
    scene_fps,
    content_fps,
    expected,
):
    frames = [np.full((2, 3, 3), value, np.uint8) for value in range(12)]
    _reader_factory(monkeypatch, frames=frames, fps=content_fps)
    source = content.VideoContentSource("content.mp4")

    actual = [
        _value(source.frame_at(_request(index, index / scene_fps)))
        for index in range(len(expected))
    ]

    assert actual == expected
    source.close()


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        pytest.param("hold", [2, 2], id="hold-last-frame"),
        pytest.param("hide", [None, None], id="hide-after-eof"),
    ],
)
def test_video_source_nonlooping_end_policies(monkeypatch, policy, expected):
    _reader_factory(monkeypatch)
    source = content.VideoContentSource("content.mp4", end_policy=policy)

    actual = [
        _value(source.frame_at(_request(0, 1.5))),
        _value(source.frame_at(_request(1, 3.0))),
    ]

    assert actual == expected


def test_video_source_loops_without_retaining_a_full_video_cache(monkeypatch):
    instances = _reader_factory(monkeypatch)
    source = content.VideoContentSource("content.mp4", end_policy="loop")

    actual = [
        _value(source.frame_at(_request(0, 0.0))),
        _value(source.frame_at(_request(1, 1.0))),
        _value(source.frame_at(_request(2, 1.5))),
        _value(source.frame_at(_request(3, 2.0))),
        _value(source.frame_at(_request(4, 3.5))),
    ]

    assert actual == [0, 2, 0, 1, 1]
    assert len(instances) == 3
    assert all(reader.released for reader in instances[:-1])
    source.close()
    assert instances[-1].released


def test_video_source_rejects_backward_timestamps_without_advancing(monkeypatch):
    _reader_factory(monkeypatch)
    source = content.VideoContentSource("content.mp4")
    assert _value(source.frame_at(_request(0, 0.5))) == 1

    with pytest.raises(ValueError, match="non-decreasing"):
        source.frame_at(_request(1, 0.4))

    assert _value(source.frame_at(_request(2, 1.0))) == 2


def test_video_source_owns_reader_and_close_is_idempotent(monkeypatch):
    instances = _reader_factory(monkeypatch)

    with content.VideoContentSource("content.mp4") as source:
        assert _value(source.frame_at(_request(0, 0.0))) == 0
    source.close()

    assert instances[0].released
    with pytest.raises(RuntimeError, match="closed"):
        source.frame_at(_request(1, 0.5))


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), True])
def test_video_source_rejects_invalid_fps_and_releases_reader(monkeypatch, fps):
    instances = _reader_factory(monkeypatch, fps=fps)

    with pytest.raises(ValueError, match="fps"):
        content.VideoContentSource("content.mp4")

    assert instances[0].released


@pytest.mark.parametrize("policy", ["hold", "loop", "hide"])
def test_video_source_rejects_an_empty_video_without_looping_forever(
    monkeypatch,
    policy,
):
    _reader_factory(monkeypatch, frames=[])
    source = content.VideoContentSource("content.mp4", end_policy=policy)

    with pytest.raises(RuntimeError, match="no decodable frames"):
        source.frame_at(_request(0, 0.0))


@pytest.mark.parametrize("policy", ["", "repeat", 1])
def test_video_source_rejects_invalid_end_policy_before_opening(monkeypatch, policy):
    opened = False

    def open_reader(path):
        nonlocal opened
        opened = True
        return _FakeReader(path)

    monkeypatch.setattr(content, "VideoReader", open_reader)

    with pytest.raises(ValueError, match="end_policy"):
        content.VideoContentSource("content.mp4", end_policy=policy)
    assert not opened


@pytest.mark.parametrize("path", ["", "   ", 42])
def test_video_source_rejects_invalid_path_before_opening(monkeypatch, path):
    opened = False

    def open_reader(value):
        nonlocal opened
        opened = True
        return _FakeReader(value)

    monkeypatch.setattr(content, "VideoReader", open_reader)

    with pytest.raises((TypeError, ValueError), match="path"):
        content.VideoContentSource(path)
    assert not opened
