"""Focused tests for shifted and Kaplan-style joint scaling laws."""

import numpy as np
import pytest

from simple_scaling_laws.laws import LawError, build_law
from simple_scaling_laws.model import raw_scale_params


def test_shifted_power_reduces_to_separable_when_shifts_are_zero():
    """Shifts nest the ordinary separable power law."""
    shifted = build_law("shifted-separable-power", ["model_size__n"], ["dataset_size__d"])
    separable = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
    log_x = np.log(np.array([[0.25, 0.5], [1.0, 2.0], [4.0, 8.0]]))
    shifted_params = np.array([1.0, 2.0, 3.0, 0.4, 0.25, 0.0, 0.0])
    separable_params = np.array([1.0, 2.0, 3.0, 0.4, 0.25])
    assert np.allclose(
        shifted.evaluate(shifted_params, log_x),
        separable.evaluate(separable_params, log_x),
    )


def test_shifted_power_has_finite_zero_scale_limit():
    """Positive shifts prevent the divergent small-scale behavior of an ordinary power law."""
    law = build_law("shifted-separable-power", ["model_size__n"], [])
    # E, A, alpha, s.
    params = np.array([1.0, 2.0, 0.5, 0.25])
    tiny = np.log(np.array([[1e-12]]))
    value = law.evaluate(params, tiny)[0]
    assert value == pytest.approx(1.0 + 2.0 * 0.25**-0.5, rel=1e-9)


def test_shifted_raw_scale_conversion():
    """A normalized shift becomes an additive shift in raw predictor units."""
    law = build_law("shifted-separable-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([1.0, 2.0, 3.0, 0.5, 0.25, 0.2, 0.5])
    raw = raw_scale_params(law, params, np.array([100.0, 16.0]))
    assert raw["A"] == pytest.approx(20.0)
    assert raw["B"] == pytest.approx(6.0)
    assert raw["s_n"] == pytest.approx(20.0)
    assert raw["s_d"] == pytest.approx(8.0)


def test_kaplan_joint_law_has_expected_limits():
    """The joint law becomes model-limited with abundant data and data-limited at huge model size."""
    law = build_law("kaplan-joint-power", ["model_size__n"], ["dataset_size__d"])
    # E + A * (N^-rho + R/D)^beta.
    params = np.array([1.0, 2.0, 0.5, 0.4, 3.0])

    abundant_data = np.log(np.array([[4.0, 1e12]]))
    observed = law.evaluate(params, abundant_data)[0]
    expected = 1.0 + 2.0 * (4.0**-0.5) ** 0.4
    assert observed == pytest.approx(expected, rel=1e-5)

    huge_model = np.log(np.array([[1e12, 4.0]]))
    observed = law.evaluate(params, huge_model)[0]
    expected = 1.0 + 2.0 * (3.0 / 4.0) ** 0.4
    assert observed == pytest.approx(expected, rel=1e-5)


def test_kaplan_raw_scale_conversion_preserves_physical_surface():
    """The reported raw parameters are algebraically equivalent to the normalized law."""
    law = build_law("kaplan-joint-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([1.0, 2.0, 0.5, 0.4, 3.0])
    reference = np.array([100.0, 16.0])
    raw = raw_scale_params(law, params, reference)

    n_raw, d_raw = 400.0, 64.0
    normalized = law.evaluate(params, np.log(np.array([[n_raw / 100.0, d_raw / 16.0]])))[0]
    raw_value = raw["E"] + raw["A"] * (n_raw ** (-raw["rho"]) + raw["R"] / d_raw) ** raw["beta"]
    assert raw_value == pytest.approx(normalized)


def test_kaplan_joint_law_requires_one_model_and_one_dataset_predictor():
    """The named Kaplan form has exactly one coordinate of each resource type."""
    with pytest.raises(LawError, match="exactly one model-size and one dataset-size"):
        build_law("kaplan-joint-power", ["model_size__n"], [])
    with pytest.raises(LawError, match="exactly one model-size and one dataset-size"):
        build_law(
            "kaplan-joint-power",
            ["model_size__n", "model_size__depth"],
            ["dataset_size__d"],
        )


def test_nonlinear_parameter_kinds_are_not_reported_as_exponents():
    """Shifts and scale ratios remain distinct from true scaling exponents."""
    shifted = build_law("shifted-separable-power", ["model_size__n"], ["dataset_size__d"])
    kaplan = build_law("kaplan-joint-power", ["model_size__n"], ["dataset_size__d"])
    assert shifted.n_exponents == 2
    assert shifted.n_nonlinear == 4
    assert kaplan.n_exponents == 2
    assert kaplan.n_nonlinear == 3
