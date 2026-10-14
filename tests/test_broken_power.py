"""Focused tests for the one-break Broken Neural Scaling Law."""

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.fitting import parameter_bounds
from simple_scaling_laws.laws import LawError, build_law
from simple_scaling_laws.model import raw_scale_params
from simple_scaling_laws.simulate import simulate_runs


def test_broken_power_reduces_to_power_when_slope_change_is_zero():
    """C1=0 nests the ordinary one-dimensional power law exactly."""
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
    with pytest.raises(LawError, match="exactly one predictor"):
        build_law("broken-power", ["model_size__n"], ["dataset_size__d"])


def test_the_break_is_confined_to_the_observed_range():
    """A break outside the scales that were run cannot be told from no break at all."""
    law = build_law("broken-power", ["model_size__n"], [])
    narrow = law.nonlinear_specs(np.log(np.array([[0.1], [1.0], [10.0]])))
    wide = law.nonlinear_specs(np.log(np.array([[1e-3], [1.0], [1e3]])))
    for specs, low, high in ((narrow, 0.1, 10.0), (wide, 1e-3, 1e3)):
        location, width = specs[2], specs[3]
        assert (location.lower, location.upper) == pytest.approx((low, high))
        assert all(location.lower < seed < location.upper for seed in location.grid)
        assert width.upper == pytest.approx(np.log(high / low) / 2)
        assert all(width.lower < seed < width.upper for seed in width.grid)


def test_a_turnover_is_recovered_on_a_wide_sweep():
    """The case the seed grid was sized for: with coarser seeds five of twelve of these went wrong."""
    truth = {"E": 1.0, "A": 2.0, "c0": 0.3, "c1": -0.6, "d": 3.0, "f": 0.8}
    sizes = tuple(10.0**k for k in range(6, 14))
    for seed in range(6):
        frame = simulate_runs(
            {"test_loss__ce": truth},
            model_sizes=sizes,
            dataset_sizes=(),
            law="broken-power",
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.02,
            eval_sd=0.03,
            seed=seed,
        )
        model = fit(frame, law="broken-power", n_draws=20, seed=seed)
        assert "lack_of_fit" not in {note.code for note in model.warnings}, seed
        assert model.exponents("test_loss__ce")["c1"] == pytest.approx(-0.6, abs=0.1), seed


def test_too_few_scales_is_an_error():
    """Six parameters on one axis: five scales identify nothing, and the fit has to say so."""
    frame = simulate_runs(
        {"test_loss__ce": {"E": 1.0, "A": 2.0, "c0": 0.15, "c1": 0.5, "d": 1.0, "f": 1.0}},
        model_sizes=tuple(10.0**k for k in range(6, 11)),
        dataset_sizes=(),
        law="broken-power",
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    errors = {
        note.code for note in fit(frame, law="broken-power", n_draws=20).warnings if note.severity == "error"
    }
    assert {"too_few_predictor_levels", "underdetermined"} <= errors


def test_a_two_predictor_experiment_is_refused_with_a_reason():
    """The law is for a sweep, and a grid has to be told so rather than silently collapsed."""
    frame = simulate_runs(
        {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}}, seed=0
    )
    with pytest.raises(LawError, match="exactly one predictor"):
        fit(frame, law="broken-power", n_draws=20)


def test_a_break_well_inside_a_wide_sweep_is_not_reported_at_its_bound():
    """Closeness to a bound on a scale parameter is relative, not a fixed fraction of a decades-wide box.

    On an eight-decade sweep the old rule flagged any break below about 0.01 of the reference scale --
    1.7 decades inside the range here -- as ``parameter_at_bound``, which this law reads as "no break".
    """
    law = build_law("broken-power", ("model_size__n",), ())
    log_x = np.log(np.logspace(-4, 4, 12))[:, None]
    truth = np.array([1.0, 2.0, 0.15, 0.5, 0.005, 1.0])
    bounds = parameter_bounds(law, law.evaluate(truth, log_x), False, log_x)
    assert bounds.at_bound(truth, law.param_names) == ()
    pinned = truth.copy()
    pinned[4] = bounds.lower[4]
    assert bounds.at_bound(pinned, law.param_names) == ("d",)
