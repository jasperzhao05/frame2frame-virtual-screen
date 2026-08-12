import numpy as np
import pytest

from frame2frame import geometry as g


def _shoelace(quad):
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_euler_roundtrip():
    for yaw, pitch, roll in [(0, 0, 0), (20, -15, 10), (-35, 25, -12), (45, 40, -30)]:
        r = g.euler_to_rotation(yaw, pitch, roll)
        assert np.allclose(g.rotation_to_euler(r), [yaw, pitch, roll], atol=1e-6)


def test_rotation_is_orthonormal():
    r = g.euler_to_rotation(17, -9, 33)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_head_rotation_is_orthonormal_and_uses_pose_convention():
    r = g.head_rotation(17, -9, 33)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r @ g.Z_AXIS, g.head_forward(17, -9), atol=1e-9)


def test_head_forward_angle_signs_match_the_public_pose_contract():
    assert g.head_forward(20, 0)[0] < 0  # positive yaw looks image-left
    assert g.head_forward(-20, 0)[0] > 0
    assert g.head_forward(0, 20)[1] < 0  # positive pitch looks image-up
    assert g.head_forward(0, -20)[1] > 0


def test_project_deproject_inverse():
    k = g.camera_matrix(800, 320, 240)
    for px, py, z in [(320, 240, 500), (410, 180, 1234), (50, 470, 77)]:
        p = g.deproject(px, py, z, k)
        assert np.allclose(g.project(p, k)[0], [px, py], atol=1e-6)


def test_geometry_keeps_its_public_keyword_parameter_names():
    k = g.camera_matrix(focal=800, cx=320, cy=240)
    point = g.deproject(px=410, py=180, depth=500, k=k)

    assert np.allclose(g.project(points_cam=point, k=k)[0], [410, 180])
    assert np.allclose(g.rotation_to_euler(r=np.eye(3)), (0, 0, 0))


def test_screen_sits_in_front_of_face():
    k = g.camera_matrix(800, 320, 240)
    face = g.deproject(320, 240, 500.0, k)
    for yaw, pitch, roll in [(0, 0, 0), (25, -10, 5), (-20, 15, -8)]:
        corners = g.gaze_plane_corners(
            face, yaw, pitch, roll, distance_world=30, screen_w_px=200, screen_h_px=100, k=k
        )
        assert corners.shape == (4, 3)
        assert corners[:, 2].mean() > face[2]


def test_screen_basis_and_center_come_from_one_rigid_head_rotation():
    k = g.camera_matrix(800, 320, 240)
    face = g.deproject(320, 240, 500.0, k)
    for yaw, pitch, roll in [
        (0, 0, 0),
        (30, 0, 0),
        (-25, 15, 9),
        (40, -20, -12),
        (0, 0, 91),
        (0, 0, 180),
    ]:
        corners = g.gaze_plane_corners(
            face,
            yaw,
            pitch,
            roll,
            distance_world=30,
            screen_w_px=200,
            screen_h_px=100,
            k=k,
        )
        u = corners[1] - corners[0]
        v = corners[3] - corners[0]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)
        rotation = g.head_rotation(yaw, pitch, roll)
        assert np.allclose(u, rotation @ g.X_AXIS)
        assert np.allclose(v, rotation @ g.Y_AXIS)
        assert np.dot(normal, g.head_forward(yaw, pitch)) == pytest.approx(1.0)
        assert np.allclose(
            corners.mean(axis=0),
            face + g.head_forward(yaw, pitch) * 30,
        )


def test_screen_quad_is_nondegenerate():
    k = g.camera_matrix(800, 320, 240)
    face = g.deproject(320, 240, 500.0, k)
    corners = g.gaze_plane_corners(
        face, 10, -5, 3, distance_world=30, screen_w_px=200, screen_h_px=100, k=k
    )
    assert _shoelace(g.project(corners, k)) > 100


def test_corner_ordering_tl_tr_br_bl():
    k = g.camera_matrix(800, 320, 240)
    face = g.deproject(320, 240, 500.0, k)
    quad = g.project(
        g.gaze_plane_corners(
            face, 0, 0, 0, distance_world=30, screen_w_px=200, screen_h_px=100, k=k
        ),
        k,
    )
    tl, tr, br, bl = quad
    assert tl[0] < tr[0] and bl[0] < br[0]  # left corners left of right corners
    assert tl[1] < bl[1] and tr[1] < br[1]  # top corners above bottom corners


def test_head_forward_is_unit_and_points_ahead():
    f = g.head_forward(0, 0)
    assert np.allclose(np.linalg.norm(f), 1.0)
    assert f[2] > 0  # looking down +Z when facing the camera
