"""Focused tests for the one-break Broken Neural Scaling Law."""

import numpy as np
import pytest

from simple_scaling_laws.laws import LawError, build_law
from simple_scaling_laws.model import raw_scale_params


def test_broken_power_reduces_to_power_when_slope_change_is_zero():
    """c1=0 nests the ordinary one-dimensional power law exactly."""
    broken = build_law("broken-power", ["model_size__n"], [])
    ordinary = build_law("separable-power", ["model_size__n"], [])
    log_x = np.log(np.array([[0.1], [1.0], [10.0]]))
    broken_params = np.array([1.0, 2.0, 0.4, 0.0, 1.0, 0.5])
    ordinary_params = np.array([1.0, 2.0, 0.4])
    assert np.allclose(broken.evaluate(broken_params, log_x), ordinary.evaluate(ordinary_params, log_x))


def test_broken_power_changes_asymptotic_log_slope():
    """The slope moves from approximately -c0 before the break to -(c0+c1) after it."""
    law = build_law("broken-power", ["model_size__n"], [])
    params = np.array([0.0, 1.0, 0.4, 0.6, 1.0, 0.1])
    x = np.array([[1e-4], [1e-3], [1e3], [1e4]])
    y = law.evaluate(params, np.log(x))
    early_slope = np.log(y[1] / y[0]) / np.log(10.0)
    late_slope = np.log(y[3] / y[2]) / np.log(10.0)
    assert early_slope == pytest.approx(-0.4, abs=0.02)
    assert late_slope == pytest.approx(-1.0, abs=0.02)


def test_negative_slope_change_can_create_a_turnover():
    """A signed post-break slope change permits genuinely non-monotonic scale behavior."""
    law = build_law("broken-power", ["model_size__n"], [])
    params = np.array([0.0, 1.0, 0.4, -0.9, 1.0, 0.1])
    x = np.array([[0.01], [1.0], [100.0]])
    y = law.evaluate(params, np.log(x))
    assert y[1] < y[0]
    assert y[2] > y[1]


def test_broken_power_raw_scale_conversion():
    """Normalization rescales the amplitude and break location without changing slopes."""
    law = build_law("broken-power", ["model_size__n"], [])
    params = np.array([1.0, 2.0, 0.4, -0.2, 3.0, 0.5])
    raw = raw_scale_params(law, params, np.array([100.0]))
    assert raw["A"] == pytest.approx(2.0 * 100.0**0.4)
    assert raw["d"] == pytest.approx(300.0)
    assert raw["c0"] == 0.4
    assert raw["c1"] == -0.2


def test_broken_power_jacobian_matches_finite_difference():
    """Break location, width, and both slopes have accurate analytic derivatives."""
    law = build_law("broken-power", ["model_size__n"], [])
    params = np.array([1.0, 2.0, 0.4, -0.2, 1.5, 0.5])
    log_x = np.log(np.array([[0.1], [1.0], [10.0]]))
    analytic = law.jacobian(params, log_x)
    numerical = np.empty_like(analytic)
    for j in range(law.n_params):
        step = 1e-6 * max(1.0, abs(params[j]))
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (law.evaluate(up, log_x) - law.evaluate(down, log_x)) / (2 * step)
    assert np.allclose(analytic, numerical, atol=1e-6, rtol=1e-5)


def test_broken_power_requires_one_predictor():
    """The conservative first BNSL implementation is intentionally one-dimensional."""
    with pytest.raises(LawError, match="exactly one scale predictor"):
        build_law("broken-power", ["model_size__n"], ["dataset_size__d"])
