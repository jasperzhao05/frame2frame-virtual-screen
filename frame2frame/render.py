"""Project and draw the virtual screen plus optional pose diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry

# These two helpers were already reachable here before texture normalization
# gained its own owner. Keep the established convenience names without
# forwarding the new private texture machinery through the renderer.
from ._textures import default_texture as default_texture
from ._textures import load_texture as load_texture
from ._textures import prepare_texture
from .config import ScreenConfig
from .pose.base import BoundingBox, FaceObservation

_CAMERA_PLANE_EPSILON = 1e-6
_MAX_PROJECTED_EXTENT_FACTOR = 1000.0
_MIN_QUAD_AREA = 1.0


@dataclass(frozen=True)
class _ScreenProjection:
    """A valid screen quad at subpixel and OpenCV contour precision."""

    quad: np.ndarray
    pixels: np.ndarray
    aspect: float


def draw_pose_axis(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    center: tuple[float, float],
    size: float,
) -> np.ndarray:
    """Head-centred RGB axes from the same basis as the virtual screen."""
    tdx, tdy = center
    rotation = geometry.head_rotation(yaw, pitch, roll)
    x_axis = rotation @ geometry.X_AXIS
    y_axis = rotation @ geometry.Y_AXIS
    forward_axis = geometry.head_forward(yaw, pitch)

    x1, y1 = np.asarray(center) + size * x_axis[:2]
    x2, y2 = np.asarray(center) + size * y_axis[:2]
    x3, y3 = np.asarray(center) + size * forward_axis[:2]

    o = (int(tdx), int(tdy))
    cv2.line(frame, o, (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.line(frame, o, (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(frame, o, (int(x3), int(y3)), (255, 0, 0), 2)
    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
    return frame


def _paste_content(
    frame: np.ndarray,
    content: object,
    quad: np.ndarray,
    alpha: float,
    *,
    fit: str = "stretch",
    target_aspect: float | None = None,
) -> None:
    """Perspective-warp into the quad's clipped ROI, leaving other pixels alone."""
    h, w = frame.shape[:2]
    prepared = prepare_texture(content)
    sh, sw = prepared.bgr.shape[:2]
    outer_destination = np.asarray(quad, dtype=np.float32)
    source, destination = _content_mapping(
        sw,
        sh,
        outer_destination,
        fit,
        target_aspect,
    )
    transform = cv2.getPerspectiveTransform(source, destination)

    # Linear interpolation can touch one pixel beyond the mathematical quad.
    # A two-pixel guard keeps the antialiased edge inside this bounded ROI and
    # avoids full-frame warp and blend allocations.
    roi_destination = outer_destination if fit == "contain" else destination
    x0 = max(0, math.floor(float(np.min(roi_destination[:, 0]))) - 2)
    y0 = max(0, math.floor(float(np.min(roi_destination[:, 1]))) - 2)
    x1 = min(w, math.ceil(float(np.max(roi_destination[:, 0]))) + 3)
    y1 = min(h, math.ceil(float(np.max(roi_destination[:, 1]))) + 3)
    if x0 >= x1 or y0 >= y1 or alpha <= 0:
        return

    local_transform = (
        np.array(
            [[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        @ transform
    )
    roi_size = (x1 - x0, y1 - y0)
    border_mode = (
        cv2.BORDER_REPLICATE if prepared.alpha is None and fit != "contain" else cv2.BORDER_CONSTANT
    )
    warped = cv2.warpPerspective(
        prepared.bgr,
        local_transform,
        roi_size,
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )
    roi = frame[y0:y1, x0:x1]
    offset = np.array([x0, y0], dtype=np.float32)

    if prepared.alpha is None and fit == "contain":
        outer_weight = _polygon_weight(roi_size, outer_destination - offset)
        _blend_into(roi, warped, outer_weight * alpha)
        return

    geometric_weight = _polygon_weight(roi_size, destination - offset)

    content_weight: np.ndarray
    if prepared.alpha is None:
        content_weight = geometric_weight * alpha
    else:
        if fit == "contain":
            outer_weight = _polygon_weight(roi_size, outer_destination - offset)
            bar_weight = np.maximum(outer_weight - geometric_weight, 0.0) * alpha
            _blend_into(roi, np.zeros_like(roi), bar_weight)
        warped_alpha = cv2.warpPerspective(
            prepared.alpha,
            local_transform,
            roi_size,
            flags=cv2.INTER_LINEAR,
        )
        bounded_alpha = (
            np.minimum(warped_alpha, geometric_weight) if fit == "cover" else warped_alpha
        )
        content_weight = bounded_alpha * alpha
    _blend_into(roi, warped, content_weight)


def _content_mapping(
    width: int,
    height: int,
    quad: np.ndarray,
    fit: str,
    target_aspect: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source and destination quads without resizing content pixels."""
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    destination = np.asarray(quad, dtype=np.float32)
    if fit == "stretch":
        return source, destination
    if fit not in {"contain", "cover"}:
        raise ValueError(f"unknown screen content fit: {fit!r}")

    screen_aspect = float(target_aspect) if target_aspect is not None else width / height
    if not math.isfinite(screen_aspect) or screen_aspect <= 0:
        raise ValueError("screen content target aspect must be finite and greater than zero")
    source_aspect = width / height

    if fit == "cover":
        if source_aspect > screen_aspect:
            crop_width = height * screen_aspect
            left = (width - crop_width) / 2.0
            source[:, 0] = [left, left + crop_width, left + crop_width, left]
        elif source_aspect < screen_aspect:
            crop_height = width / screen_aspect
            top = (height - crop_height) / 2.0
            source[:, 1] = [top, top, top + crop_height, top + crop_height]
        return source, destination

    if source_aspect > screen_aspect:
        visible_fraction = screen_aspect / source_aspect
        margin = (1.0 - visible_fraction) / 2.0
        uv = np.array(
            [[0, margin], [1, margin], [1, 1 - margin], [0, 1 - margin]],
            dtype=np.float32,
        )
    else:
        visible_fraction = source_aspect / screen_aspect
        margin = (1.0 - visible_fraction) / 2.0
        uv = np.array(
            [[margin, 0], [1 - margin, 0], [1 - margin, 1], [margin, 1]],
            dtype=np.float32,
        )
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    plane_transform = cv2.getPerspectiveTransform(unit, destination)
    destination = cv2.perspectiveTransform(uv[None, ...], plane_transform)[0]
    return source, destination


def _polygon_weight(size: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Rasterize destination-space coverage without a second perspective warp."""
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    shift = 8
    fixed = np.rint(np.asarray(polygon) * (1 << shift)).astype(np.int32)
    cv2.fillConvexPoly(mask, fixed, 255, lineType=cv2.LINE_AA, shift=shift)
    return mask.astype(np.float32) * (1.0 / 255.0)


def _blend_into(background: np.ndarray, foreground: np.ndarray, weight: np.ndarray) -> None:
    """Blend into an uint8 ROI using OpenCV's vectorized implementation."""
    cv2.blendLinear(foreground, background, weight, 1.0 - weight, dst=background)


def _validated_opacity(opacity: object) -> float:
    if (
        not isinstance(opacity, (int, float))
        or isinstance(opacity, bool)
        or not math.isfinite(opacity)
    ):
        raise ValueError("screen opacity must be a finite number")
    if not 0 <= opacity <= 1:
        raise ValueError("screen opacity must be between 0 and 1")
    return float(opacity)


def _project_screen(
    frame: np.ndarray,
    observation: FaceObservation,
    cfg: ScreenConfig,
) -> _ScreenProjection | None:
    """Project one valid screen quad, or return ``None`` for unsafe geometry."""
    h, w = frame.shape[:2]
    focal = cfg.focal_length or float(max(w, h))
    k = geometry.camera_matrix(focal, w / 2.0, h / 2.0)

    tdx, tdy = observation.center
    face_size = max(1.0, float(observation.size))
    z_face = focal * cfg.depth_scale / face_size
    face_center_cam = geometry.deproject(tdx, tdy, z_face, k)

    distance_world = cfg.distance_mul * cfg.depth_scale
    screen_w_px = max(cfg.min_size_px, cfg.width_mul * face_size)
    screen_h_px = max(cfg.min_size_px, cfg.height_mul * face_size)

    pose = observation.pose
    corners = geometry.gaze_plane_corners(
        face_center_cam,
        pose.yaw,
        pose.pitch,
        pose.roll,
        distance_world=distance_world,
        screen_w_px=screen_w_px,
        screen_h_px=screen_h_px,
        k=k,
    )
    # A pinhole projection is undefined on or behind the camera plane.  Skip
    # such extreme poses instead of feeding infinities or an inverted quad to
    # OpenCV's perspective transform.
    if not np.isfinite(corners).all() or np.any(corners[:, 2] <= _CAMERA_PLANE_EPSILON):
        return None
    # Keep the quad at subpixel precision for the warp; rounding to whole
    # pixels here made the screen step visibly even after smoothing.
    quad = geometry.project(corners, k).astype(np.float32)
    if not np.isfinite(quad).all():
        return None
    # Values this far outside the image are not a useful visible projection
    # and risk overflowing the int32 contour API at near-camera singularities.
    if np.max(np.abs(quad)) > max(w, h) * _MAX_PROJECTED_EXTENT_FACTOR:
        return None
    area = cv2.contourArea(quad)
    if not math.isfinite(area) or area < _MIN_QUAD_AREA:
        return None
    return _ScreenProjection(
        quad,
        np.rint(quad).astype(np.int32),
        screen_w_px / screen_h_px,
    )


def _composite_screen(
    frame: np.ndarray,
    projection: _ScreenProjection,
    cfg: ScreenConfig,
    content: object | None,
    opacity: float,
) -> None:
    """Composite texture content or the fallback fill inside a valid quad."""
    alpha = cfg.alpha * opacity
    if content is not None:
        if cfg.content_fit == "stretch":
            _paste_content(frame, content, projection.quad, alpha)
            return
        _paste_content(
            frame,
            content,
            projection.quad,
            alpha,
            fit=cfg.content_fit,
            target_aspect=projection.aspect,
        )
        return

    overlay = frame.copy()
    cv2.fillPoly(overlay, [projection.pixels], (50, 50, 50))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_screen_border(
    frame: np.ndarray,
    projection: _ScreenProjection,
    cfg: ScreenConfig,
    opacity: float,
) -> None:
    """Draw the configured border with the same dropout opacity as the screen."""
    if cfg.border_thickness <= 0:
        return
    if opacity == 1:
        cv2.polylines(
            frame,
            [projection.pixels],
            True,
            cfg.border_color,
            cfg.border_thickness,
        )
        return

    overlay = frame.copy()
    cv2.polylines(
        overlay,
        [projection.pixels],
        True,
        cfg.border_color,
        cfg.border_thickness,
    )
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_virtual_screen(
    frame: np.ndarray,
    observation: FaceObservation,
    cfg: ScreenConfig,
    content: object | None = None,
    *,
    opacity: float = 1.0,
) -> np.ndarray:
    """Project the screen plane for one face observation and paste content onto it."""
    opacity = _validated_opacity(opacity)
    if opacity == 0:
        return frame

    projection = _project_screen(frame, observation, cfg)
    if projection is None:
        return frame

    _composite_screen(frame, projection, cfg, content, opacity)
    _draw_screen_border(frame, projection, cfg, opacity)
    return frame
