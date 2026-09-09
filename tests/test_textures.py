import numpy as np
import pytest

from frame2frame import _textures as textures


def test_prepare_texture_preserves_left_to_right_order_and_extracts_alpha():
    texture = np.array(
        [[[1, 2, 3, 0], [4, 5, 6, 255]]],
        dtype=np.uint8,
    )

    prepared = textures.prepare_texture(texture)

    assert prepared.bgr.tolist() == [[[1, 2, 3], [4, 5, 6]]]
    assert prepared.alpha.tolist() == [[0.0, 1.0]]
    assert textures.prepare_texture(prepared) is prepared


@pytest.mark.parametrize(
    ("bgr", "alpha"),
    [
        pytest.param(
            np.ones((2, 2, 3), np.uint16),
            np.ones((2, 2), np.float32),
            id="non-uint8-bgr",
        ),
        pytest.param(
            np.ones((2, 2, 3), np.uint8),
            np.ones((2, 2), np.float64),
            id="non-float32-alpha",
        ),
        pytest.param(
            np.ones((2, 2, 3), np.uint8),
            np.full((2, 2), np.nan, np.float32),
            id="nonfinite-alpha",
        ),
        pytest.param(
            np.ones((2, 2, 3), np.uint8),
            np.full((2, 2), 1.1, np.float32),
            id="alpha-above-one",
        ),
    ],
)
def test_prepared_texture_rejects_invalid_internal_invariants(bgr, alpha):
    with pytest.raises(ValueError, match="prepared texture"):
        textures._PreparedTexture(bgr, alpha)


def test_prepare_texture_accepts_grayscale():
    prepared = textures.prepare_texture(np.array([[10, 20]], np.uint8))

    assert prepared.bgr.shape == (1, 2, 3)
    assert prepared.bgr[0, :, 0].tolist() == [10, 20]
    assert prepared.alpha is None


def test_uint8_bgr_uses_the_zero_copy_opaque_fast_path():
    texture = np.zeros((20, 30, 3), np.uint8)

    prepared = textures.prepare_texture(texture)

    assert np.shares_memory(prepared.bgr, texture)
    assert prepared.alpha is None


def test_prepare_texture_accepts_default_integer_numpy_arrays():
    texture = np.array([[[1, 128, 255, 128]]])

    prepared = textures.prepare_texture(texture)

    assert prepared.bgr[0, 0].tolist() == [1, 128, 255]
    assert prepared.alpha[0, 0] == pytest.approx(128 / 255)


def test_prepare_texture_scales_uint16_bgra_without_wrapping():
    texture = np.array([[[0, 32_768, 65_535, 32_768]]], dtype=np.uint16)

    prepared = textures.prepare_texture(texture)

    assert prepared.bgr.dtype == np.uint8
    assert prepared.bgr[0, 0].tolist() == [0, 128, 255]
    assert prepared.alpha[0, 0] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    ("texture", "expected"),
    [
        pytest.param(
            np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32),
            [0, 128, 255],
            id="unit-range",
        ),
        pytest.param(
            np.array([[[0.0, 128.0, 255.0]]], dtype=np.float32),
            [0, 128, 255],
            id="byte-range",
        ),
    ],
)
def test_prepare_texture_normalizes_float_color_ranges(texture, expected):
    prepared = textures.prepare_texture(texture)

    assert prepared.bgr.dtype == np.uint8
    assert prepared.bgr[0, 0].tolist() == expected


@pytest.mark.parametrize(
    "texture",
    [
        pytest.param(
            np.array([[[float("nan"), 0.0, 1.0]]], dtype=np.float32),
            id="nan",
        ),
        pytest.param(
            np.array([[[float("inf"), 0.0, 1.0]]], dtype=np.float32),
            id="infinity",
        ),
        pytest.param(
            np.array([[[-0.1, 0.0, 1.0]]], dtype=np.float32),
            id="negative",
        ),
        pytest.param(
            np.array([[[0.0, 1.0, 256.0]]], dtype=np.float32),
            id="above-255",
        ),
    ],
)
def test_prepare_texture_rejects_nonfinite_or_out_of_range_colors(texture):
    with pytest.raises(ValueError, match="texture color|texture colors"):
        textures.prepare_texture(texture)


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(-0.1, id="negative"),
        pytest.param(256.0, id="above-255"),
    ],
)
def test_prepare_texture_rejects_invalid_float_alpha(alpha):
    texture = np.array([[[0.0, 0.0, 0.0, alpha]]], dtype=np.float32)

    with pytest.raises(ValueError, match="alpha"):
        textures.prepare_texture(texture)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(np.empty((0, 0, 3), np.uint8), id="empty"),
        pytest.param(np.zeros((2, 2, 2), np.uint8), id="two-channel"),
        pytest.param(np.zeros((2,), np.uint8), id="one-dimensional"),
    ],
)
def test_prepare_texture_rejects_invalid_images(content):
    with pytest.raises(ValueError, match="texture"):
        textures.prepare_texture(content)
