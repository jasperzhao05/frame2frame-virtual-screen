"""Pinhole-camera geometry for placing a head-locked virtual screen.

Everything here is plain numpy so the math can be unit-tested without OpenCV or
a model backend. Convention: the camera looks down +Z, image x grows to the
right and y grows downwards.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def camera_matrix(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def euler_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Tait-Bryan rotation R = Rx(pitch) @ Ry(yaw) @ Rz(roll), angles in degrees."""
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rx @ ry @ rz


def rotation_to_euler(r: ArrayLike) -> tuple[float, float, float]:
    """Inverse of euler_to_rotation; returns (yaw, pitch, roll) in degrees.

    Degenerate at yaw = +/-90 (gimbal lock), which the head-pose range never
    reaches in practice.
    """
    matrix = np.asarray(r, dtype=float)
    yaw = np.degrees(np.arcsin(np.clip(matrix[0, 2], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-matrix[0, 1], matrix[0, 0]))
    pitch = np.degrees(np.arctan2(-matrix[1, 2], matrix[2, 2]))
    return float(yaw), float(pitch), float(roll)


def project(points_cam: ArrayLike, k: ArrayLike) -> np.ndarray:
    """Project Nx3 camera-space points to Nx2 pixels (no lens distortion)."""
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    matrix = np.asarray(k, dtype=float)
    uvw = pts @ matrix.T
    return uvw[:, :2] / uvw[:, 2:3]


def deproject(px: float, py: float, depth: float, k: ArrayLike) -> np.ndarray:
    """Back-project a pixel at a given depth into camera space."""
    matrix = np.asarray(k, dtype=float)
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    return np.array([(px - cx) / fx * depth, (py - cy) / fy * depth, float(depth)])


def head_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Map the neutral head basis into the OpenCV-like camera frame.

    At neutral pose, local ``+X`` points image-right, local ``+Y`` points
    image-down, and the face looks along local ``-Z`` toward the camera.
    Positive yaw looks image-left, positive pitch looks up, and positive roll
    tilts the local right axis image-down.
    """
    return euler_to_rotation(yaw, -pitch, roll)


def head_forward(yaw: float, pitch: float) -> np.ndarray:
    """Unit ray the head looks along, in camera space."""
    return head_rotation(yaw, pitch, 0.0) @ -Z_AXIS


def gaze_plane_corners(
    face_center_cam: ArrayLike,
    yaw: float,
    pitch: float,
    roll: float,
    *,
    distance_world: float,
    screen_w_px: float,
    screen_h_px: float,
    k: ArrayLike,
) -> np.ndarray:
    """Four screen corners (tl, tr, br, bl) in camera space.

    The plane floats ``distance_world`` ahead of the face along the head's
    forward ray and is tilted by the head rotation. Half extents are evaluated
    at the plane depth so the screen keeps a stable on-image size.
    """
    face_center_cam = np.asarray(face_center_cam, dtype=float)
    center = face_center_cam + head_forward(yaw, pitch) * distance_world

    camera = np.asarray(k, dtype=float)
    focal = camera[0, 0]
    zs = center[2]
    half_w = (screen_w_px / 2.0) / focal * zs
    half_h = (screen_h_px / 2.0) / focal * zs

    # The basis and placement ray share one renderer rotation. With texture
    # coordinates ordered right then down, cross(u, v) points away from the
    # visible face of the plane, so the gaze direction is -cross(u, v).
    r = head_rotation(yaw, pitch, roll)
    u = r @ X_AXIS
    v = r @ Y_AXIS

    return np.array(
        [
            center - u * half_w - v * half_h,  # top-left
            center + u * half_w - v * half_h,  # top-right
            center + u * half_w + v * half_h,  # bottom-right
            center - u * half_w + v * half_h,  # bottom-left
        ]
    )
