import numpy as np
import pytest

from frame2frame.config import FilterConfig
from frame2frame.filters import (
    FIRAttitudeFilter,
    KalmanAttitudeFilter,
    _wrap_step,
    create_filter,
)


def _run(filt, signal):
    return np.array([filt.update(v, 0.0, 0.0)[0] for v in signal])


def test_fir_attenuates_noise_but_keeps_low_freq():
    fps, n = 30.0, 600
    t = np.arange(n) / fps
    low = 10 * np.sin(2 * np.pi * 0.3 * t)  # well inside the passband
    noise = np.random.default_rng(0).normal(0, 4, n)
    out = _run(create_filter(fps, FilterConfig(kind="fir")), low + noise)

    gd = FIRAttitudeFilter(fps, FilterConfig()).group_delay
    aligned, ref = out[gd:], low[: n - gd]
    # what remains after smoothing tracks the clean low-frequency component
    assert np.std(aligned - ref) < np.std(noise)
    assert out.std() < (low + noise).std()


def test_fir_passband_gain_is_unity():
    fps, n = 30.0, 600
    t = np.arange(n) / fps
    low = 10 * np.sin(2 * np.pi * 0.3 * t)
    filt = create_filter(fps, FilterConfig(kind="fir"))
    out = _run(filt, low)
    gd = filt.group_delay
    aligned, ref = out[gd:], low[: n - gd]
    assert np.allclose(aligned[150:], ref[150:], atol=1.0)  # amplitude preserved


def test_group_delay_matches_tap_count():
    filt = FIRAttitudeFilter(30.0, FilterConfig())
    assert filt.group_delay == (filt.n - 1) // 2


@pytest.mark.parametrize(
    ("kind", "group_delay"),
    [
        pytest.param("fir", None, id="fir"),
        pytest.param("kalman", 0, id="kalman"),
        pytest.param("oneeuro", 0, id="oneeuro"),
        pytest.param("none", 0, id="passthrough"),
    ],
)
def test_all_filters_support_the_tracker_operations(kind, group_delay):
    cfg = (
        FilterConfig(kind=kind, smooth_translation=False)
        if kind == "kalman"
        else FilterConfig(kind=kind)
    )
    filt = create_filter(30.0, cfg)

    assert isinstance(filt.group_delay, int)
    if group_delay is not None:
        assert filt.group_delay == group_delay
    assert len(filt.update(1.0, 2.0, 3.0, dt=1 / 30)) == 3
    assert len(filt.update_position(cx=4.0, cy=5.0, size=6.0, dt=1 / 30)) == 3


def test_fir_and_passthrough_accept_timing_without_changing_values():
    fir_without_dt = create_filter(30.0, FilterConfig(kind="fir"))
    fir_with_dt = create_filter(30.0, FilterConfig(kind="fir"))
    assert fir_with_dt.update(1.0, 2.0, 3.0, dt=0.5) == pytest.approx(
        fir_without_dt.update(1.0, 2.0, 3.0)
    )

    passthrough = create_filter(30.0, FilterConfig(kind="none"))
    assert passthrough.update(1.0, 2.0, 3.0, dt=0.5) == (1.0, 2.0, 3.0)


def test_oneeuro_settles_on_a_step():
    filt = create_filter(30.0, FilterConfig(kind="oneeuro"))
    out = (0, 0, 0)
    for _ in range(200):
        out = filt.update(10.0, 0.0, 0.0)
    assert abs(out[0] - 10.0) < 0.5


def test_oneeuro_derivative_uses_consecutive_raw_samples():
    filt = create_filter(10.0, FilterConfig(kind="oneeuro", beta=1.0))
    filt.update(0.0, 0.0, 0.0)
    filt.update(10.0, 0.0, 0.0)
    derivative_after_step = filt._dx["yaw"]

    filt.update(10.0, 0.0, 0.0)

    expected_decay = derivative_after_step * (1.0 - filt._alpha(filt.d_cutoff))
    assert filt._dx["yaw"] == pytest.approx(expected_decay)


def test_oneeuro_uses_the_supplied_wall_clock_timestep():
    filt = create_filter(10.0, FilterConfig(kind="oneeuro", beta=1.0))
    assert filt.uses_timestamps is True
    filt.update(0.0, 0.0, 0.0, dt=0.2)
    filt.update(10.0, 0.0, 0.0, dt=0.2)

    expected_raw_derivative = 50.0
    expected_filtered = expected_raw_derivative * filt._alpha(filt.d_cutoff, 0.2)
    assert filt._dx["yaw"] == pytest.approx(expected_filtered)


@pytest.mark.parametrize(
    "dt",
    [
        pytest.param(0, id="zero"),
        pytest.param(-0.1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_oneeuro_rejects_invalid_wall_clock_timestep(dt):
    filt = create_filter(10.0, FilterConfig(kind="oneeuro"))

    with pytest.raises(ValueError, match="dt"):
        filt.update(0.0, 0.0, 0.0, dt=dt)


def _experiment_kalman_reference(signal, fps=30.0, dts=None):
    """Frozen NumPy equations from the registered BIWI A100 experiment."""
    state = None
    covariance = None
    previous = None
    output = []
    acceleration_variance = 100.0**2
    measurement_variance = 1.0**2
    intervals = [1.0 / fps] * len(signal) if dts is None else dts
    for raw, dt in zip(signal, intervals):
        value = float(raw)
        if previous is not None:
            value = previous + (value - previous + 180.0) % 360.0 - 180.0
        previous = value
        if state is None:
            state = np.array([value, 0.0], dtype=np.float64)
            covariance = np.diag([measurement_variance, acceleration_variance])
        else:
            transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
            acceleration = np.array([0.5 * dt * dt, dt], dtype=np.float64)
            process = np.outer(acceleration, acceleration) * acceleration_variance
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            innovation_variance = covariance[0, 0] + measurement_variance
            gain = covariance[:, 0] / innovation_variance
            state += gain * (value - state[0])
            covariance -= np.outer(gain, covariance[0, :])
        output.append(float(state[0]))
    return np.asarray(output)


def test_kalman_matches_registered_a100_experiment_at_nominal_dt():
    signal = np.array([0.0, 4.0, 7.0, 5.0, -2.0, 178.0, -179.0, -175.0])
    filt = create_filter(
        30.0,
        FilterConfig(kind="kalman", smooth_translation=False),
    )

    actual = np.array([filt.update(value, 0.0, 0.0)[0] for value in signal])

    assert actual == pytest.approx(_experiment_kalman_reference(signal), abs=1e-12)


def test_kalman_uses_actual_dt_and_has_no_fixed_group_delay():
    signal = np.array([0.0, 10.0, 13.0, -4.0, 8.0])
    dts = [0.01, 0.2, 0.04, 0.12, 0.03]
    filt = create_filter(30.0, FilterConfig(kind="kalman", smooth_translation=False))

    actual = np.array([filt.update(value, 0.0, 0.0, dt=dt)[0] for value, dt in zip(signal, dts)])

    assert actual == pytest.approx(
        _experiment_kalman_reference(signal, dts=dts),
        abs=1e-12,
    )
    assert filt.group_delay == 0
    assert filt.uses_timestamps is True


def test_kalman_wraps_resets_and_leaves_position_unchanged():
    filt = create_filter(30.0, FilterConfig(kind="kalman", smooth_translation=False))
    filt.update(179.0, 0.0, 0.0)
    wrapped = filt.update(-179.0, 0.0, 0.0)[0]
    assert 179.0 < wrapped < 182.0
    assert filt.update_position(40.0, 50.0, 60.0, dt=0.2) == (40.0, 50.0, 60.0)

    filt.reset()
    assert filt.update(-25.0, 2.0, 3.0) == (-25.0, 2.0, 3.0)


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_kalman_rejects_invalid_wall_clock_timestep(dt):
    filt = create_filter(30.0, FilterConfig(kind="kalman", smooth_translation=False))

    with pytest.raises(ValueError, match="Kalman dt"):
        filt.update(0.0, 0.0, 0.0, dt=dt)


def test_kalman_direct_constructor_uses_registered_defaults():
    filt = KalmanAttitudeFilter(30.0)

    assert filt.group_delay == 0
    assert filt.update(1.0, 2.0, 3.0) == (1.0, 2.0, 3.0)


def test_none_passes_through():
    filt = create_filter(30.0, FilterConfig(kind="none"))
    assert filt.update(3.0, 4.0, 5.0) == (3.0, 4.0, 5.0)
    spaced = create_filter(30.0, FilterConfig(kind="  NONE  "))
    assert spaced.update(3.0, 4.0, 5.0) == (3.0, 4.0, 5.0)


def test_unknown_filter_kind_raises():
    with pytest.raises(ValueError):
        create_filter(30.0, FilterConfig(kind="bogus"))


@pytest.mark.parametrize(
    "fps",
    [
        pytest.param(0, id="zero"),
        pytest.param(-30, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(True, id="boolean"),
    ],
)
def test_filters_reject_invalid_frame_rates(fps):
    with pytest.raises(ValueError, match="fps must be"):
        create_filter(fps, FilterConfig(kind="fir"))
    with pytest.raises(ValueError, match="fps must be"):
        create_filter(fps, FilterConfig(kind="oneeuro"))
    with pytest.raises(ValueError, match="fps must be"):
        create_filter(fps, FilterConfig(kind="kalman", smooth_translation=False))
    with pytest.raises(ValueError, match="fps must be"):
        create_filter(fps, FilterConfig(kind="none"))


def test_fir_rejects_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        create_filter(5.0, FilterConfig(kind="fir", cutoff_hz=2.5))


def test_fir_accepts_low_frame_rate_with_wide_transition():
    filt = create_filter(6.0, FilterConfig(kind="fir", cutoff_hz=2.5, transition_hz=5.0))
    assert all(np.isfinite(filt.update(1.0, 2.0, 3.0)))


def test_oneeuro_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="min_cutoff"):
        create_filter(30.0, FilterConfig(kind="oneeuro", min_cutoff=0))
    with pytest.raises(ValueError, match="beta"):
        create_filter(30.0, FilterConfig(kind="oneeuro", beta=-0.1))


def test_create_filter_uses_the_same_configuration_validation_as_pipeline():
    with pytest.raises(ValueError, match="ripple_db"):
        create_filter(30.0, FilterConfig(kind="fir", ripple_db=7.0))
    with pytest.raises(ValueError, match="roll_cutoff_scale"):
        create_filter(30.0, FilterConfig(kind="none", roll_cutoff_scale=1.1))


def test_filter_config_owns_smooth_translation_validation():
    with pytest.raises(ValueError, match="smooth_translation"):
        FilterConfig(smooth_translation=1).validate()


def test_wrap_step_crosses_the_seam():
    assert _wrap_step(179, None) == 179
    assert _wrap_step(-179, 179) == 181
    assert _wrap_step(179, -179) == -181
    assert _wrap_step(5, 3) == 5
