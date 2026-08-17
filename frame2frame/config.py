"""Runtime configuration objects.

Screen scale defaults preserve the values hand-tuned in the original prototype;
tracking defaults add an explicit short-gap hold/fade policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

ColorChannel = Union[int, float]
BorderColor = tuple[ColorChannel, ColorChannel, ColorChannel]


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> None:
    """Raise a useful error instead of passing NaN/inf into OpenCV or SciPy."""
    number = _finite_number(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number")
    outside_range = number < 0 if allow_zero else number <= 0
    if outside_range:
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {relation}")


def _optional_path(name: str, value: str | None, *, suffix: str) -> None:
    """Validate an optional non-empty path and its required file suffix."""
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path or None")
    if not Path(value).suffix:
        raise ValueError(f"{name} filename must include {suffix}")


def _same_path(first: str, second: str) -> bool:
    """Compare paths lexically and by inode when both already exist."""
    left = Path(first).expanduser().resolve()
    right = Path(second).expanduser().resolve()
    if left == right:
        return True
    if left.exists() and right.exists():
        try:
            return left.samefile(right)
        except OSError:
            return False
    return False


def _distinct_paths(
    first_name: str,
    first: str | None,
    second_name: str,
    second: str | None,
) -> None:
    if first is not None and second is not None and _same_path(first, second):
        raise ValueError(f"{first_name} and {second_name} must be different paths")


def _validate_distinct_paths(paths: tuple[tuple[str, str | None], ...]) -> None:
    """Require every populated path role to identify a different file.

    Each newly introduced role is checked against earlier roles. Besides being
    easier to extend than a hand-written pair list, this preserves the existing
    validation order and therefore the first error users see.
    """
    previous: list[tuple[str, str | None]] = []
    for name, path in paths:
        for previous_name, previous_path in previous:
            _distinct_paths(previous_name, previous_path, name, path)
        previous.append((name, path))


def _integer(name: str, value: object, *, minimum: int, description: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be {description}")


def _booleans(values: tuple[tuple[str, object], ...]) -> None:
    for name, value in values:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")


def _input_path(input_path: object, webcam: object) -> str | None:
    file_input = input_path if isinstance(input_path, str) and input_path.strip() else None
    has_webcam = webcam is not None
    if (file_input is not None) == has_webcam:
        raise ValueError("set exactly one of input or webcam")
    if has_webcam:
        _integer("webcam", webcam, minimum=0, description="a non-negative integer index")
    return file_input


def _validate_audio(preserve_audio: bool, input_path: str | None, output: str | None) -> None:
    if not preserve_audio:
        return
    if input_path is None:
        raise ValueError("preserve_audio requires a file input, not a webcam")
    if output is None:
        raise ValueError("preserve_audio requires an output path")


def _color_channel(value: object) -> bool:
    number = _finite_number(value)
    return number is not None and 0 <= number <= 255


@dataclass
class ScreenConfig:
    distance_mul: float = 2.0  # screen distance, in depth_scale units
    width_mul: float = 4.0  # screen width, in face-size units
    height_mul: float = 2.0  # screen height, in face-size units
    depth_scale: float = 6.0  # face depth = focal * depth_scale / face_size
    min_size_px: float = 40.0
    alpha: float = 0.8  # blend strength of the pasted content
    focal_length: float | None = None  # defaults to max(width, height)
    texture_path: str | None = None
    border_color: BorderColor = (0, 200, 255)
    border_thickness: int = 0

    def validate(self) -> None:
        for name in ("distance_mul", "width_mul", "height_mul", "depth_scale", "min_size_px"):
            _positive_finite(f"screen.{name}", getattr(self, name))
        _positive_finite("screen.alpha", self.alpha, allow_zero=True)
        if self.alpha > 1:
            raise ValueError("screen.alpha must be between 0 and 1")
        if self.focal_length is not None:
            _positive_finite("screen.focal_length", self.focal_length)
        if self.texture_path is not None and (
            not isinstance(self.texture_path, str) or not self.texture_path.strip()
        ):
            raise ValueError("screen.texture_path must be a non-empty path or None")
        _integer(
            "screen.border_thickness",
            self.border_thickness,
            minimum=0,
            description="a non-negative integer",
        )
        if (
            not isinstance(self.border_color, tuple)
            or len(self.border_color) != 3
            or not all(_color_channel(channel) for channel in self.border_color)
        ):
            raise ValueError("screen.border_color must contain three values between 0 and 255")


@dataclass
class FilterConfig:
    kind: str = "fir"  # fir | oneeuro | none
    smooth_translation: bool = True  # also smooth the face centre/size, not just angles

    # FIR (Kaiser window low-pass)
    cutoff_hz: float = 2.5
    transition_hz: float = 5.0
    ripple_db: float = 60.0
    pitch_cutoff_scale: float = 0.5  # pitch is smoothed harder than yaw
    roll_cutoff_scale: float = 0.1  # roll hardest of all

    # One Euro
    min_cutoff: float = 1.0
    beta: float = 0.3
    d_cutoff: float = 1.0

    def validate(self, fps: float | None = None) -> None:
        kind = self.kind.strip().lower() if isinstance(self.kind, str) else ""
        if kind not in {"fir", "oneeuro", "none", "off", "passthrough"}:
            raise ValueError(f"unknown filter kind: {self.kind!r}")
        _booleans((("filter.smooth_translation", self.smooth_translation),))

        for name in (
            "cutoff_hz",
            "transition_hz",
            "ripple_db",
            "min_cutoff",
            "d_cutoff",
        ):
            _positive_finite(f"filter.{name}", getattr(self, name))
        for name in ("pitch_cutoff_scale", "roll_cutoff_scale"):
            value = getattr(self, name)
            _positive_finite(f"filter.{name}", value)
            if value > 1:
                raise ValueError(f"filter.{name} must be no greater than 1")
        _positive_finite("filter.beta", self.beta, allow_zero=True)
        if self.ripple_db < 8:
            raise ValueError("filter.ripple_db must be at least 8 dB")

        if fps is not None:
            _positive_finite("fps", fps)
            if kind == "fir":
                nyquist = fps / 2.0
                if self.cutoff_hz >= nyquist:
                    raise ValueError(
                        "FIR cutoff frequencies must be below the source Nyquist frequency "
                        f"({nyquist:g} Hz)"
                    )


@dataclass
class PipelineConfig:
    input: str | None = None
    webcam: int | None = None
    output: str | None = "output/processed.mp4"
    backend: str = "mediapipe"
    backend_kwargs: dict[str, object] = field(default_factory=dict)

    filter: FilterConfig = field(default_factory=FilterConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)

    # Offline FIR only: hold frames back by the filter's group delay so the
    # smoothed pose is composited onto the frame it belongs to. Ignored for
    # webcam/display, where added latency would be felt.
    compensate_delay: bool = True

    draw_screen: bool = True
    draw_axis: bool = False
    draw_bbox: bool = False

    display: bool = False
    plot_path: str | None = "output/angle_processed.png"

    # Plotting is optional diagnostics, not a reason to retain an unbounded
    # history for a webcam or a multi-hour recording.
    max_plot_samples: int = 10_000

    # OpenCV does not carry input audio into its encoded output. When enabled,
    # the pipeline writes video to a temporary sibling file, then asks ffmpeg
    # to copy the original audio into a second temporary file and atomically
    # installs the completed result.
    preserve_audio: bool = False

    # Keep the screen stable across the short detector misses common in video,
    # then fade it before a sustained miss resets the temporal filter. These
    # are wall-clock durations so behavior does not change with source FPS.
    # Kept at the end so existing positional construction is not shifted.
    dropout_hold_seconds: float = 0.2
    dropout_reset_seconds: float = 0.5

    def validate(self, fps: float | None = None) -> None:
        input_path = _input_path(self.input, self.webcam)
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.backend_kwargs, dict):
            raise ValueError("backend_kwargs must be a dictionary")
        _optional_path("output", self.output, suffix="a container suffix")
        _optional_path("plot_path", self.plot_path, suffix="an image suffix")
        _integer(
            "max_plot_samples",
            self.max_plot_samples,
            minimum=1,
            description="a positive integer",
        )
        _positive_finite("dropout_hold_seconds", self.dropout_hold_seconds, allow_zero=True)
        _positive_finite("dropout_reset_seconds", self.dropout_reset_seconds)
        if self.dropout_hold_seconds > self.dropout_reset_seconds:
            raise ValueError("dropout_hold_seconds must not exceed dropout_reset_seconds")
        _booleans(
            (
                ("compensate_delay", self.compensate_delay),
                ("draw_screen", self.draw_screen),
                ("draw_axis", self.draw_axis),
                ("draw_bbox", self.draw_bbox),
                ("display", self.display),
                ("preserve_audio", self.preserve_audio),
            )
        )

        _validate_distinct_paths(
            (
                ("input", input_path),
                ("output", self.output),
                ("plot_path", self.plot_path),
                ("screen.texture_path", self.screen.texture_path),
            )
        )
        _validate_audio(self.preserve_audio, input_path, self.output)

        self.screen.validate()
        self.filter.validate(fps)
