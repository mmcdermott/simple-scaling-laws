"""Compute-optimal allocation from a fitted law: correctness, edge cases and calibration."""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from simple_scaling_laws import fit
from simple_scaling_laws.cost import cost_model
from simple_scaling_laws.model import PredictionError
from simple_scaling_laws.simulate import simulate_runs

PREDICTORS = ("model_size__n_params", "dataset_size__n_tokens")
MODEL_SIZES = (1e6, 1e7, 1e8, 1e9)
DATASET_SIZES = (1e7, 1e8, 1e9, 1e10)
#: Unequal exponents, so the optimum leans toward model size and the frontier is not symmetric.
TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}
COST = cost_model("power", k=6.0, a=1.0, b=1.0)


def _runs(seed=0, noise=0.01, truth=TRUTH, law="separable-power", **kwargs):
    return simulate_runs(
        {"test_loss__ce": truth},
        model_sizes=MODEL_SIZES,
        dataset_sizes=DATASET_SIZES,
        predictors=PREDICTORS,
        law=law,
        run_sd=noise,
        eval_sd=noise,
        seed=seed,
        **kwargs,
    )


def _true_optimum(model, budget, cost=COST, truth=TRUTH):
    """The optimal model size under the generating law, found by a scalar search on the iso-cost curve."""
    reference = np.array([model.observed_domain[p]["reference"] for p in model.fitted_predictors])
    vector = np.array([truth[name] for name in model.law.param_names])

    def loss(log_n):
        n = np.exp(log_n)
        point = np.log(np.array([[n, float(cost.dataset_size_for(budget, n))]])) - np.log(reference)
        return float(model.law.evaluate(vector, point)[0])

    return float(np.exp(minimize_scalar(loss, bounds=(np.log(1e3), np.log(1e14)), method="bounded").x))


def test_noiseless_fits_find_the_true_optimum():
    model = fit(_runs(noise=0.0, runs_per_config=1, evaluations_per_run=1), n_draws=20, seed=0)
    frontier = model.optimal_allocation([1e15, 1e17, 1e19], COST)
    for budget, found in zip((1e15, 1e17, 1e19), frontier["model_size__n_params__median"], strict=True):
        assert found == pytest.approx(_true_optimum(model, budget), rel=1e-3)
    spent = COST(frontier["model_size__n_params__median"], frontier["dataset_size__n_tokens__median"])
    assert spent == pytest.approx([1e15, 1e17, 1e19], rel=1e-6)


def test_the_allocation_exponent_matches_the_separable_closed_form():
    """``N* ∝ C ** (beta / (alpha + beta))`` for the separable law with cost ``k N D``."""
    model = fit(_runs(noise=0.0, runs_per_config=1, evaluations_per_run=1), n_draws=20, seed=0)
    exponents = model.allocation_exponents([1e14, 1e16, 1e18], COST)
    alpha, beta = TRUTH["alpha"], TRUTH["beta"]
    assert exponents["model_size"]["median"] == pytest.approx(beta / (alpha + beta), abs=1e-3)
    assert exponents["dataset_size"]["median"] == pytest.approx(alpha / (alpha + beta), abs=1e-3)


def test_a_law_without_an_interior_optimum_says_so():
    """Along any iso-cost line the multiplicative law is monotone, so its optimum runs to the edge."""
    model = fit(_runs(), law="multiplicative-power", n_draws=60, seed=0)
    with pytest.warns(UserWarning, match="no_interior_optimum"):
        frontier = model.optimal_allocation([1e17], COST)
    assert frontier["p_no_interior_optimum"][0] > 0.9


def test_an_optimum_beyond_the_grid_is_flagged_as_extrapolation():
    model = fit(_runs(), n_draws=60, seed=0)
    with pytest.warns(UserWarning, match="optimum_outside_observed_domain"):
        frontier = model.optimal_allocation([1e24], COST)
    assert frontier["domain"][0] == "extrapolation"
    assert frontier["extrapolation_distance"][0] > 0


def test_a_metric_that_improves_with_scale_is_maximized():
    """Higher is better for AUROC, so the optimum is its maximum, and agrees with the loss's here."""
    metric = {"E": 2.0, "A": -0.5, "alpha": 0.3, "B": -0.375, "beta": 0.4}
    runs = simulate_runs(
        {"test_loss__ce": TRUTH, "test_metric__auroc": metric},
        model_sizes=MODEL_SIZES,
        dataset_sizes=DATASET_SIZES,
        predictors=PREDICTORS,
        run_sd=0.0,
        eval_sd=0.0,
        runs_per_config=1,
        evaluations_per_run=1,
    )
    model = fit(runs, n_draws=20, seed=0)
    frontier = model.optimal_allocation([1e17], COST, targets=["test_loss__ce", "test_metric__auroc"])
    loss_n, metric_n = frontier["model_size__n_params__median"]
    assert metric_n == pytest.approx(loss_n, rel=1e-3)


def test_a_law_over_one_predictor_cannot_allocate():
    sweep = simulate_runs({"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3}}, dataset_sizes=())
    model = fit(sweep, n_draws=20, seed=0)
    with pytest.raises(PredictionError, match="one model-size predictor against one dataset-size"):
        model.optimal_allocation([1e17], COST)


def test_frontier_intervals_cover_the_true_optimum_beyond_the_grid():
    """Calibration of the frontier itself, at a budget a hundred times the largest one run."""
    budget, hits = 1e21, 0
    n_replications = 15
    for seed in range(n_replications):
        model = fit(
            _runs(seed=seed, noise=0.02, runs_per_config=2, evaluations_per_run=4), n_draws=120, seed=seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = model.optimal_allocation([budget], COST).row(0, named=True)
        truth = _true_optimum(model, budget)
        hits += row["model_size__n_params__q025"] <= truth <= row["model_size__n_params__q975"]
    assert hits / n_replications >= 0.8
