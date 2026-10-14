"""Focused tests for the bounded rational small-data scaling law."""

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.laws import build_law
from simple_scaling_laws.model import raw_scale_params
from simple_scaling_laws.simulate import simulate_runs


def test_bounded_rational_approaches_both_limits():
    """The M4 link with its shape exponent at one approaches I at tiny scale and E at huge scale."""
    law = build_law("bounded-rational-power", ["model_size__n"], [])
    # E, I, A, alpha.
    params = np.array([1.0, 4.0, 2.0, 0.5])
    tiny = law.evaluate(params, np.log(np.array([[1e-12]])))[0]
    huge = law.evaluate(params, np.log(np.array([[1e12]])))[0]
    assert tiny == pytest.approx(4.0, rel=1e-6)
    assert huge == pytest.approx(1.0, rel=1e-5)


def test_bounded_rational_stays_between_limits_for_positive_g():
    """Positive scaling pressure makes the response a convex combination of E and I."""
    law = build_law("bounded-rational-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([0.8, 2.5, 1.5, 0.7, 0.4, 0.3])
    log_x = np.log(np.array([[0.01, 0.1], [1.0, 1.0], [100.0, 10.0]]))
    values = law.evaluate(params, log_x)
    assert np.all(values > 0.8)
    assert np.all(values < 2.5)


def test_bounded_rational_matches_m4_relation_at_unit_shape():
    """The fitted response exactly satisfies (f-E)/(I-f) = g."""
    law = build_law("bounded-rational-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 0.25])
    x = np.array([[0.25, 2.0], [1.0, 1.0], [4.0, 8.0]])
    values = law.evaluate(params, np.log(x))
    g = 2.0 * x[:, 0] ** -0.5 + 4.0 * x[:, 1] ** -0.25
    assert np.allclose((values - 1.0) / (3.0 - values), g)


def test_bounded_rational_raw_scale_conversion():
    """The g coefficients absorb predictor normalization exactly."""
    law = build_law("bounded-rational-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 0.25])
    reference = np.array([100.0, 16.0])
    raw = raw_scale_params(law, params, reference)
    assert raw["A"] == pytest.approx(20.0)
    assert raw["B"] == pytest.approx(8.0)
    assert raw["E"] == 1.0
    assert raw["I"] == 3.0


def test_bounded_rational_jacobian_matches_finite_difference():
    """Both g-amplitude and exponent derivatives are analytic and accurate."""
    law = build_law("bounded-rational-power", ["model_size__n"], ["dataset_size__d"])
    params = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 0.25])
    log_x = np.log(np.array([[0.25, 0.5], [1.0, 2.0], [4.0, 8.0]]))
    analytic = law.jacobian(params, log_x)
    numerical = np.empty_like(analytic)
    for j in range(law.n_params):
        step = 1e-6 * max(1.0, abs(params[j]))
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (law.evaluate(up, log_x) - law.evaluate(down, log_x)) / (2 * step)
    assert np.allclose(analytic, numerical, atol=1e-6, rtol=1e-5)


def test_on_data_that_never_bend_the_unidentified_plateau_is_reported_as_ill_conditioned():
    """The positive case for ``ill_conditioned_fit``: ``I`` and the weights trade off along a flat valley."""
    truth = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}
    for seed in range(3):
        frame = simulate_runs(
            {"test_loss__ce": truth},
            model_sizes=(1e6, 1e7, 1e8, 1e9),
            dataset_sizes=(1e3, 1e4, 1e5, 1e6),
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.01,
            eval_sd=0.02,
            seed=seed,
        )
        model = fit(frame, law="bounded-rational-power", n_draws=40, seed=seed)
        flagged = [note for note in model.warnings if note.code == "ill_conditioned_fit"]
        assert flagged, seed
        assert "I" in flagged[0].details["parameters"], seed
        assert flagged[0].details["condition_number"] is None or flagged[0].details["condition_number"] > 1e3
