import numpy as np
import pytest

from frame2frame import video


class FakeCapture:
    def __init__(self, *, opened=True, frames=(), values=None):
        self.opened = opened
        self.frames = list(frames)
        self.values = values or {}
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeCvWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def test_video_reader_falls_back_from_nonfinite_metadata(monkeypatch):
    frame = np.zeros((12, 20, 3), np.uint8)
    capture = FakeCapture(
        frames=[frame],
        values={
            video.cv2.CAP_PROP_FPS: float("nan"),
            video.cv2.CAP_PROP_FRAME_COUNT: float("nan"),
        },
    )
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture)

    with video.VideoReader("clip.mp4") as reader:
        assert reader.fps == 30.0
        assert reader.frame_count == 0
        assert reader.size == (20, 12)
        assert next(reader) is frame

    assert capture.released


@pytest.mark.parametrize(
    ("rotation", "operation"),
    [
        pytest.param(90, video.cv2.ROTATE_90_CLOCKWISE, id="90-clockwise"),
        pytest.param(180, video.cv2.ROTATE_180, id="180"),
        pytest.param(
            270,
            video.cv2.ROTATE_90_COUNTERCLOCKWISE,
            id="270-counterclockwise",
        ),
    ],
)
def test_video_reader_applies_rotation_to_every_frame(monkeypatch, rotation, operation):
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    orientation = video.cv2.CAP_PROP_ORIENTATION_META
    capture = FakeCapture(frames=[frame, frame], values={orientation: rotation})
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture)

    expected = video.cv2.rotate(frame, operation)
    with video.VideoReader("rotated.mp4") as reader:
        frames = list(reader)
        assert reader.size == (expected.shape[1], expected.shape[0])

    assert all(np.array_equal(actual, expected) for actual in frames)
    orientation_auto = getattr(video.cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        assert (orientation_auto, 0) in capture.set_calls


def test_reader_releases_capture_when_initial_frame_is_unreadable(monkeypatch):
    capture = FakeCapture(opened=True)
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(RuntimeError, match="no decodable frames"):
        video.VideoReader("empty.mp4")

    assert capture.released


def test_webcam_releases_capture_when_initial_grab_fails(monkeypatch):
    capture = FakeCapture(opened=True)
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda index: capture)

    with pytest.raises(RuntimeError, match="initial frame"):
        video.WebcamReader(0)

    assert capture.released


@pytest.mark.parametrize(
    "fps",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_writer_rejects_invalid_fps_before_opening_file(tmp_path, fps):
    with pytest.raises(ValueError):
        video.VideoWriter(tmp_path / "bad.mp4", fps, (20, 12))


@pytest.mark.parametrize(
    "size",
    [
        pytest.param((0, 12), id="zero-width"),
        pytest.param((20, -1), id="negative-height"),
        pytest.param((20.0, 12), id="noninteger-width"),
        pytest.param((20,), id="missing-height"),
    ],
)
def test_writer_rejects_invalid_size(tmp_path, size):
    with pytest.raises(ValueError, match="size"):
        video.VideoWriter(tmp_path / "bad.mp4", 30.0, size)


def test_writer_releases_failed_opencv_writer(monkeypatch, tmp_path):
    writer = FakeCvWriter(opened=False)
    monkeypatch.setattr(video.cv2, "VideoWriter", lambda *args: writer)

    with pytest.raises(RuntimeError, match="cannot open"):
        video.VideoWriter(tmp_path / "bad.mp4", 30.0, (20, 12))

    assert writer.released


def test_writer_checks_frame_size_and_release_is_idempotent(monkeypatch, tmp_path):
    writer = FakeCvWriter()
    monkeypatch.setattr(video.cv2, "VideoWriter", lambda *args: writer)
    wrapped = video.VideoWriter(tmp_path / "out.mp4", 30.0, (20, 12))

    with pytest.raises(ValueError, match="frame size"):
        wrapped.write(np.zeros((10, 20, 3), np.uint8))
    wrapped.release()
    wrapped.release()

    assert writer.released


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(np.zeros((12, 20), np.uint8), id="grayscale"),
        pytest.param(np.zeros((12, 20, 4), np.uint8), id="four-channel"),
        pytest.param(np.zeros((12, 20, 3), np.float32), id="float-bgr"),
    ],
)
def test_writer_rejects_non_bgr_uint8_frames(monkeypatch, tmp_path, frame):
    writer = FakeCvWriter()
    monkeypatch.setattr(video.cv2, "VideoWriter", lambda *args: writer)
    wrapped = video.VideoWriter(tmp_path / "out.mp4", 30.0, (20, 12))

    with pytest.raises(ValueError):
        wrapped.write(frame)

    wrapped.release()
    assert not writer.frames
