"""Internal texture loading and normalization for screen rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class _PreparedTexture:
    """Static, camera-facing texture channels prepared once per pipeline run."""

    bgr: np.ndarray
    alpha: np.ndarray

    def __post_init__(self) -> None:
        valid_color = (
            isinstance(self.bgr, np.ndarray)
            and self.bgr.dtype == np.uint8
            and self.bgr.ndim == 3
            and self.bgr.shape[2] == 3
        )
        if not valid_color:
            raise ValueError("prepared texture color must be a uint8 BGR array")
        valid_alpha = (
            isinstance(self.alpha, np.ndarray)
            and self.alpha.dtype == np.float32
            and self.alpha.shape == self.bgr.shape[:2]
        )
        if not valid_alpha:
            raise ValueError("prepared texture alpha must be a matching float32 plane")
        if self.bgr.size == 0:
            raise ValueError("prepared texture must not be empty")
        if not np.isfinite(self.alpha).all() or np.any((self.alpha < 0) | (self.alpha > 1)):
            raise ValueError("prepared texture alpha must contain finite values in [0, 1]")


def default_texture(width: int = 640, height: int = 360) -> np.ndarray:
    """Create a self-contained placeholder so demos need no asset files."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = xx / max(width - 1, 1)
    gy = yy / max(height - 1, 1)
    image = np.empty((height, width, 3), np.uint8)
    image[..., 0] = np.clip(60 + 150 * gy, 0, 255)  # B
    image[..., 1] = np.clip(40 + 120 * gx, 0, 255)  # G
    image[..., 2] = np.clip(90 + 120 * (1 - gx), 0, 255)  # R

    grid = image.copy()
    step = max(20, width // 16)
    for x in range(0, width, step):
        cv2.line(grid, (x, 0), (x, height), (235, 235, 235), 1, cv2.LINE_AA)
    for y in range(0, height, step):
        cv2.line(grid, (0, y), (width, y), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.addWeighted(grid, 0.35, image, 0.65, 0, image)

    cv2.drawMarker(
        image,
        (width // 2, height // 2),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        40,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), (255, 255, 255), 2)
    return image


def load_texture(path: str | Path) -> np.ndarray:
    """Load a grayscale, BGR, or BGRA texture without changing its channels."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read texture: {path}")
    return image


def _scaled_values(values: object, *, alpha: bool) -> tuple[np.ndarray, float]:
    """Return float32 image values and the scale that represents full intensity."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float32), 1.0

    label = "texture alpha" if alpha else "texture colors"
    if np.issubdtype(array.dtype, np.integer):
        if np.min(array) < 0:
            value_label = "texture alpha" if alpha else "texture color"
            raise ValueError(f"{value_label} values must be non-negative")
        observed_maximum = int(np.max(array))
        is_default_integer = (
            np.issubdtype(array.dtype, np.signedinteger) and array.dtype.itemsize > 2
        )
        if is_default_integer and alpha and observed_maximum <= 1:
            scale = 1.0
        elif is_default_integer and observed_maximum <= 255:
            scale = 255.0
        else:
            scale = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32), scale

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must contain only finite values")
        minimum, maximum = float(np.min(array)), float(np.max(array))
        if minimum < 0 or maximum > 255:
            raise ValueError(f"floating-point {label} must be in [0, 1] or [0, 255]")
        scale = 1.0 if maximum <= 1.0 else 255.0
        return array.astype(np.float32), scale

    raise ValueError(f"{label} must use a boolean, integer, or floating-point dtype")


def _color_to_uint8(values: object) -> np.ndarray:
    """Convert common image dtypes without wrapping or silently darkening."""
    scaled, scale = _scaled_values(values, alpha=False)
    return np.rint(scaled * (255.0 / scale)).astype(np.uint8)


def _alpha_to_float(values: object) -> np.ndarray:
    scaled, scale = _scaled_values(values, alpha=True)
    return np.clip(scaled / scale, 0.0, 1.0)


def prepare_texture(content: object) -> _PreparedTexture:
    """Normalize a texture once instead of once per video frame.

    Texture coordinates use the conventional top-left, top-right,
    bottom-right, bottom-left ordering. Mirroring here would reverse readable
    text and asymmetric content even when the head pose is neutral.
    """
    if isinstance(content, _PreparedTexture):
        return content
    image = np.asarray(content)
    if image.size == 0:
        raise ValueError("texture must not be empty")

    if image.ndim == 2:
        channels = 1
        color = image
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        channels = image.shape[2]
        color = image[..., 0] if channels == 1 else image[..., :3]
    else:
        raise ValueError("texture must be grayscale, BGR, or BGRA")

    color = _color_to_uint8(color)
    bgr = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR) if channels == 1 else color
    alpha = (
        _alpha_to_float(image[..., 3]) if channels == 4 else np.ones(image.shape[:2], np.float32)
    )

    return _PreparedTexture(
        np.ascontiguousarray(bgr),
        np.ascontiguousarray(alpha),
    )
