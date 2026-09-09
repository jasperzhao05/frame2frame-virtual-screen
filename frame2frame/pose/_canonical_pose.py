"""Recover head rotation in the renderer's own pinhole-camera coordinates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import cv2
import numpy as np

from ..geometry import camera_matrix

_LANDMARK_COUNT = 468

# Canonical face: x-right, y-up, z-toward-viewer.
# Renderer camera: x-right, y-down, z-into-scene.
_CANONICAL_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


@lru_cache(maxsize=1)
def _canonical_vertices() -> np.ndarray:
    """Load the attributed MediaPipe vertices in Face Landmarker order."""
    text = (
        resources.files(__package__)
        .joinpath("data", "canonical_face_vertices.txt")
        .read_text(encoding="utf-8")
    )
    rows = [
        tuple(float(value) for value in line.split()[1:4])
        for line in text.splitlines()
        if line.startswith("v ")
    ]
    vertices = np.asarray(rows, dtype=np.float64)
    if vertices.shape != (_LANDMARK_COUNT, 3):
        raise RuntimeError("canonical face data must contain 468 three-dimensional vertices")
    vertices.setflags(write=False)
    return vertices


def _focal_length(value: float | None, width: int, height: int) -> float:
    if value is None:
        return float(max(width, height))
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("focal_length must be a positive finite number or None")
    return float(value)


def solve_renderer_rotation(
    landmarks: np.ndarray,
    frame_shape: Sequence[int],
    *,
    focal_length: float | None = None,
) -> np.ndarray | None:
    """Fit the canonical face and return its rotation in renderer coordinates.

    Face Landmarker estimates image landmarks under its own virtual-camera
    geometry. Fitting those landmarks again with the renderer's camera keeps
    screen placement and pose interpretation on one pinhole model.
    """
    if len(frame_shape) < 2:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if width <= 0 or height <= 0:
        return None

    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < _LANDMARK_COUNT or points.shape[1] < 2:
        return None
    image_points = np.ascontiguousarray(points[:_LANDMARK_COUNT, :2])
    if not np.isfinite(image_points).all():
        return None

    focal = _focal_length(focal_length, width, height)
    intrinsic = camera_matrix(focal, width / 2.0, height / 2.0)
    solved, rotation_vector, _ = cv2.solvePnP(
        _canonical_vertices(),
        image_points,
        intrinsic,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        return None
    canonical_rotation, _ = cv2.Rodrigues(rotation_vector)
    renderer_rotation = canonical_rotation @ _CANONICAL_TO_CAMERA
    if not np.isfinite(renderer_rotation).all():
        return None
    return renderer_rotation
