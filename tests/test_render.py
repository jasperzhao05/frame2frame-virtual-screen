import cv2
import numpy as np
import pytest

from frame2frame import _textures as textures
from frame2frame import render
from frame2frame.config import ScreenConfig
from frame2frame.pose.base import FaceObservation, HeadPose


def _full_frame_reference(frame, content, quad, alpha):
    if content.ndim == 3 and content.shape[2] == 4:
        bgr = content[:, :, :3]
        source_alpha = content[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr = content
        source_alpha = np.ones(content.shape[:2], np.float32)
    sh, sw = bgr.shape[:2]
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    matrix = cv2.getPerspectiveTransform(src, quad)
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)
    warped_alpha = (
        cv2.warpPerspective(source_alpha, matrix, (w, h), flags=cv2.INTER_LINEAR) * alpha
    )[:, :, None]
    blended = (
        frame.astype(np.float32) * (1 - warped_alpha) + warped.astype(np.float32) * warped_alpha
    )
    return blended.astype(frame.dtype)


def test_roi_paste_matches_full_frame_reference():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    content = rng.integers(0, 256, (45, 80, 4), dtype=np.uint8)
    quad = np.float32([[70.3, 45.4], [171.7, 50.2], [164.4, 128.8], [61.1, 120.6]])
    expected = _full_frame_reference(frame.copy(), content, quad, 0.73)
    actual = frame.copy()

    render._paste_content(actual, textures.prepare_texture(content), quad, 0.73)

    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    # blendLinear rounds where the former NumPy expression truncated.  The
    # efficient path is visually equivalent but may differ by one code value.
    assert difference.max() <= 1
    assert np.count_nonzero(difference) < frame.size * 0.11


def test_paste_warps_only_the_quad_roi(monkeypatch):
    frame = np.zeros((300, 500, 3), np.uint8)
    content = np.full((30, 50, 3), 200, np.uint8)
    quad = np.float32([[20, 30], [90, 30], [90, 80], [20, 80]])
    sizes = []
    original = render.cv2.warpPerspective

    def recording_warp(*args, **kwargs):
        sizes.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(render.cv2, "warpPerspective", recording_warp)
    render._paste_content(frame, content, quad, 1.0)

    # Opaque content needs one color warp; destination-space coverage is
    # rasterized directly instead of paying for a second perspective warp.
    assert len(sizes) == 1
    assert all(width < 500 and height < 300 for width, height in sizes)


def test_zero_alpha_and_offscreen_quad_leave_frame_unchanged():
    initial = np.full((30, 40, 3), 17, np.uint8)
    texture = np.full((5, 8, 3), 200, np.uint8)

    zero_alpha = initial.copy()
    render._paste_content(
        zero_alpha,
        texture,
        np.float32([[2, 2], [20, 2], [20, 20], [2, 20]]),
        0.0,
    )
    offscreen = initial.copy()
    render._paste_content(
        offscreen,
        texture,
        np.float32([[-30, -30], [-20, -30], [-20, -20], [-30, -20]]),
        1.0,
    )

    assert np.array_equal(zero_alpha, initial)
    assert np.array_equal(offscreen, initial)


def test_asymmetric_texture_keeps_readable_left_to_right_orientation():
    frame = np.zeros((80, 120, 3), np.uint8)
    texture = np.zeros((20, 40, 3), np.uint8)
    texture[:, :20] = (0, 0, 255)
    texture[:, 20:] = (0, 255, 0)
    quad = np.float32([[20, 20], [100, 20], [100, 60], [20, 60]])

    render._paste_content(frame, texture, quad, 1.0)

    assert frame[40, 35, 2] > frame[40, 35, 1]  # red remains on the left
    assert frame[40, 85, 1] > frame[40, 85, 2]  # green remains on the right


def test_contain_preserves_aspect_by_insetting_the_destination_quad():
    outer = np.float32([[0, 0], [200, 0], [200, 100], [0, 100]])

    source, destination = render._content_mapping(100, 100, outer, "contain", 2.0)

    np.testing.assert_array_equal(source, np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]))
    np.testing.assert_allclose(
        destination,
        np.float32([[50, 0], [150, 0], [150, 100], [50, 100]]),
        atol=1e-4,
    )


def test_cover_preserves_aspect_by_cropping_source_coordinates():
    outer = np.float32([[0, 0], [200, 0], [200, 100], [0, 100]])

    source, destination = render._content_mapping(100, 100, outer, "cover", 2.0)

    np.testing.assert_allclose(
        source,
        np.float32([[0, 25], [100, 25], [100, 75], [0, 75]]),
    )
    np.testing.assert_array_equal(destination, outer)


def test_contain_renders_centered_content_and_black_bars_inside_the_plane():
    frame = np.full((100, 160, 3), 80, np.uint8)
    content = np.full((40, 40, 3), (10, 20, 200), np.uint8)
    quad = np.float32([[20, 20], [140, 20], [140, 80], [20, 80]])

    render._paste_content(frame, content, quad, 1.0, fit="contain", target_aspect=2.0)

    assert frame[50, 30].max() <= 1  # left bar
    assert frame[50, 130].max() <= 1  # right bar
    np.testing.assert_allclose(frame[50, 80], (10, 20, 200), atol=1)
    np.testing.assert_array_equal(frame[10, 10], (80, 80, 80))


def test_cover_crops_source_coordinates_instead_of_squashing_the_full_image():
    frame = np.zeros((100, 160, 3), np.uint8)
    content = np.empty((100, 100, 3), np.uint8)
    content[:25] = (255, 0, 0)
    content[25:75] = (0, 255, 0)
    content[75:] = (0, 0, 255)
    quad = np.float32([[20, 20], [140, 20], [140, 80], [20, 80]])

    render._paste_content(frame, content, quad, 1.0, fit="cover", target_aspect=2.0)

    assert frame[28, 80, 1] > 240
    assert frame[72, 80, 1] > 240
    assert frame[50, 80, 0] < 10
    assert frame[50, 80, 2] < 10


def test_cover_does_not_bleed_more_than_two_pixels_outside_the_quad():
    frame = np.full((120, 180, 3), 17, np.uint8)
    content = np.full((80, 80, 3), 230, np.uint8)
    quad = np.float32([[38.4, 28.2], [151.7, 35.6], [142.1, 94.4], [29.3, 87.8]])

    render._paste_content(frame, content, quad, 1.0, fit="cover", target_aspect=2.0)

    changed = np.argwhere(np.any(frame != 17, axis=2))
    distances = [cv2.pointPolygonTest(quad, (float(x), float(y)), True) for y, x in changed]
    assert min(distances) >= -2.0


def test_projection_records_actual_aspect_after_minimum_size_clamping():
    frame = np.zeros((120, 160, 3), np.uint8)
    observation = FaceObservation(HeadPose(0, 0, 0), (80, 60), 5, (75, 55, 85, 65))

    projection = render._project_screen(
        frame,
        observation,
        ScreenConfig(width_mul=4.0, height_mul=2.0, min_size_px=40.0),
    )

    assert projection is not None
    assert projection.aspect == pytest.approx(1.0)


def test_pose_axis_uses_the_forward_gaze_direction(monkeypatch):
    lines = []
    monkeypatch.setattr(
        render.cv2,
        "line",
        lambda _frame, start, end, color, thickness: lines.append((start, end, color, thickness)),
    )

    render.draw_pose_axis(np.zeros((100, 100, 3), np.uint8), 25, 0, 0, (50, 50), 20)

    blue_axis = next(line for line in lines if line[2] == (255, 0, 0))
    assert blue_axis[1][0] < blue_axis[0][0]


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [
        (-45, -30, -30),
        (-45, -30, 30),
        (-45, 30, -30),
        (-45, 30, 30),
        (45, -30, -30),
        (45, -30, 30),
        (45, 30, -30),
        (45, 30, 30),
    ],
)
def test_normal_operating_envelope_projects_to_a_finite_quad(yaw, pitch, roll):
    frame = np.zeros((180, 320, 3), np.uint8)
    observation = FaceObservation(
        HeadPose(yaw, pitch, roll),
        (160, 90),
        28,
        (132, 62, 188, 118),
    )

    projection = render._project_screen(frame, observation, ScreenConfig())

    assert projection is not None
    assert np.isfinite(projection.quad).all()
    assert np.isfinite(projection.pixels).all()


def test_screen_with_any_corner_behind_camera_is_a_safe_noop(monkeypatch):
    frame = np.full((80, 120, 3), 17, np.uint8)
    observation = FaceObservation(HeadPose(0, 0, 0), (60, 40), 20, (40, 20, 80, 60))
    corners = np.array(
        [[-1, -1, 10], [1, -1, 10], [1, 1, -1], [-1, 1, 10]],
        dtype=float,
    )
    monkeypatch.setattr(render.geometry, "gaze_plane_corners", lambda *args, **kwargs: corners)

    actual = render.draw_virtual_screen(
        frame.copy(),
        observation,
        ScreenConfig(),
        np.full((10, 10, 3), 200, np.uint8),
    )

    assert np.array_equal(actual, frame)


def test_near_camera_projection_too_large_for_safe_contours_is_a_noop(monkeypatch):
    frame = np.full((80, 120, 3), 17, np.uint8)
    observation = FaceObservation(HeadPose(0, 0, 0), (60, 40), 20, (40, 20, 80, 60))
    corners = np.array(
        [
            [-1, -1, 2e-6],
            [1, -1, 2e-6],
            [1, 1, 2e-6],
            [-1, 1, 2e-6],
        ],
        dtype=float,
    )
    monkeypatch.setattr(render.geometry, "gaze_plane_corners", lambda *args, **kwargs: corners)

    actual = render.draw_virtual_screen(
        frame.copy(),
        observation,
        ScreenConfig(),
        np.full((10, 10, 3), 200, np.uint8),
    )

    assert np.array_equal(actual, frame)


def test_screen_opacity_scales_texture_compositing(monkeypatch):
    frame = np.zeros((80, 120, 3), np.uint8)
    observation = FaceObservation(HeadPose(0, 0, 0), (60, 40), 20, (40, 20, 80, 60))
    captured = {}
    monkeypatch.setattr(
        render,
        "_paste_content",
        lambda frame, content, quad, alpha: captured.update(alpha=alpha),
    )

    render.draw_virtual_screen(
        frame,
        observation,
        ScreenConfig(alpha=0.8),
        np.full((10, 10, 3), 200, np.uint8),
        opacity=0.25,
    )

    assert captured["alpha"] == pytest.approx(0.2)
