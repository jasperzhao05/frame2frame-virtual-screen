"""Video and webcam IO.

The reader exists mostly to paper over two things OpenCV gets wrong often
enough to matter: phone clips carry a rotation flag that some FFmpeg builds
ignore (so faces come out sideways), and the reported frame size doesn't always
match what you actually decode. We read the rotation tag, apply it ourselves,
and take the real frame size from the first decoded frame.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, SupportsFloat, SupportsIndex, Union, cast

import cv2
import numpy as np

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class _Capture(Protocol):
    """The small OpenCV capture surface shared by production and test readers."""

    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


def _finite_float(value: object) -> float | None:
    try:
        number = float(cast(Union[str, SupportsFloat, SupportsIndex], value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_float(value: object) -> float | None:
    number = _finite_float(value)
    return number if number is not None and number > 0 else None


def _usable_fps(value: object, fallback: object = 30.0) -> float:
    """Return a finite positive FPS, tolerating broken capture metadata."""
    fps = _positive_float(value)
    if fps is not None:
        return fps
    if fallback is None:
        raise ValueError("fps must be finite and greater than zero")
    fps = _positive_float(fallback)
    if fps is None:
        raise ValueError("fallback fps must be finite and greater than zero")
    return fps


def _reported_frame_count(capture: _Capture) -> int:
    count = _finite_float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return max(0, int(count)) if count is not None else 0


def _decoded_frame(capture: _Capture) -> np.ndarray | None:
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


class _CaptureReader:
    """Shared capture lifecycle and iteration for file and webcam readers."""

    def __init__(
        self,
        source: str | int,
        *,
        open_error: BaseException,
        empty_error: BaseException,
        size_error: BaseException,
    ) -> None:
        self.cap = cv2.VideoCapture(source)
        try:
            if not self.cap.isOpened():
                raise open_error
            self._prepare_capture()
            frame = _decoded_frame(self.cap)
            if frame is None:
                raise empty_error
            frame = self._transform_frame(frame)
            self._pending: np.ndarray | None = frame
            self.height, self.width = frame.shape[:2]
            if self.width <= 0 or self.height <= 0:
                raise size_error
        except BaseException:
            self.cap.release()
            raise

    def _prepare_capture(self) -> None:
        """Read metadata and configure the capture before its first frame."""
        pass

    def _transform_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        decoded = _decoded_frame(self.cap)
        if decoded is None:
            raise StopIteration
        return self._transform_frame(decoded)

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> _CaptureReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class VideoReader(_CaptureReader):
    """Decode a file using its post-rotation first frame as the output shape.

    Construction opens the source and decodes one frame. Iteration returns that
    cached frame first, so deriving the real shape never drops source content.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        super().__init__(
            self.path,
            open_error=FileNotFoundError(f"cannot open video: {self.path}"),
            empty_error=RuntimeError(f"video contains no decodable frames: {self.path}"),
            size_error=RuntimeError(f"video reported an invalid frame size: {self.path}"),
        )

    def _prepare_capture(self) -> None:
        self.fps = _usable_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = _reported_frame_count(self.cap)
        self._rotation = self._read_rotation()

        # Rotation metadata is handled consistently in _transform_frame.
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None:
            self.cap.set(orientation_auto, 0)

    def _read_rotation(self) -> int:
        orientation_meta = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if orientation_meta is None:
            return 0
        rotation = _finite_float(self.cap.get(orientation_meta))
        return int(round(rotation)) % 360 if rotation is not None else 0

    def _transform_frame(self, frame: np.ndarray) -> np.ndarray:
        operation = _ROTATIONS.get(self._rotation)
        return cv2.rotate(frame, operation) if operation is not None else frame


class WebcamReader(_CaptureReader):
    """Open a webcam and cache its first decoded frame for shape discovery."""

    def __init__(self, index: int = 0, fps: float = 30.0) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("webcam index must be a non-negative integer")
        self._fallback_fps = fps
        super().__init__(
            index,
            open_error=RuntimeError(f"cannot open webcam {index}"),
            empty_error=RuntimeError(f"cannot read an initial frame from webcam {index}"),
            size_error=RuntimeError(f"webcam {index} reported an invalid frame size"),
        )

    def _prepare_capture(self) -> None:
        self.fps = _usable_fps(self.cap.get(cv2.CAP_PROP_FPS), self._fallback_fps)
        self.frame_count = 0


class VideoWriter:
    """Validate and encode fixed-size uint8 BGR frames."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int] | list[int],
        fourcc: str = "mp4v",
    ) -> None:
        fps = _usable_fps(fps, fallback=None)
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in size
            )
        ):
            raise ValueError("video size must be two positive integers")
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise ValueError("fourcc must contain exactly four characters")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(output)
        self.size = tuple(size)
        self._released = False
        # The OpenCV stubs omit this module-level factory even though every
        # supported runtime exposes it.
        fourcc_code = vars(cv2)["VideoWriter_fourcc"](*fourcc)
        self.writer = cv2.VideoWriter(self.path, fourcc_code, fps, self.size)
        if not self.writer.isOpened():
            self.writer.release()
            self._released = True
            raise RuntimeError(f"cannot open VideoWriter for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("cannot write to a released VideoWriter")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("video frames must have HxWx3 BGR shape")
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"frame size must match the configured writer size {self.size}")
        if frame.dtype != np.uint8:
            raise ValueError("video frames must use uint8 BGR pixels")
        self.writer.write(frame)

    def release(self) -> None:
        if not self._released:
            self.writer.release()
            self._released = True

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
