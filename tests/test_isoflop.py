"""IsoFLOP profile fits: recovery, calibration, diagnostics and the comparison with a fitted law."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize_scalar

from simple_scaling_laws import fit
from simple_scaling_laws.cost import cost_model
from simple_scaling_laws.isoflop import IsoflopError, fit_isoflop
from simple_scaling_laws.laws import build_law
from simple_scaling_laws.simulate import simulate_runs

COST = cost_model("power", k=6.0, a=1.0, b=1.0)
PREDICTORS = ("model_size__n_params", "dataset_size__n_tokens")
#: Unequal exponents, so each profile is lopsided and the parabola's bias would show.
TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}
BUDGETS = (1e15, 1e16, 1e17, 1e18, 1e19)


def _design(budgets=BUDGETS, points=6, half_width=0.6, shift=0.0):
    """Model sizes spread evenly in log around a guess at the optimum, at each budget."""
    configs = []
    for budget in budgets:
        centre = np.sqrt(budget / 6.0 / 10.0) * 10**shift
        for n in centre * np.logspace(-half_width, half_width, points):
            configs.append((float(n), float(COST.dataset_size_for(budget, n))))
    return configs


def _runs(configs, seed=0, noise=0.02, truth=TRUTH, **kwargs):
    return simulate_runs(
        {"test_loss__ce": truth},
        configurations=configs,
        predictors=PREDICTORS,
        run_sd=noise,
        eval_sd=1.5 * noise,
        seed=seed,
        **kwargs,
    )


def _true_optimum(configs, budget, truth=TRUTH):
    law = build_law("separable-power", (PREDICTORS[0],), (PREDICTORS[1],))
    reference = np.exp(np.log(np.array(configs)).mean(axis=0))
    vector = np.array([truth[name] for name in law.param_names])

    def loss(log_n):
        n = np.exp(log_n)
        x = np.log(np.array([[n, float(COST.dataset_size_for(budget, n))]])) - np.log(reference)
        return float(law.evaluate(vector, x)[0])

    return float(np.exp(minimize_scalar(loss, bounds=(np.log(1e2), np.log(1e16)), method="bounded").x))


def test_noiseless_profiles_find_each_optimum_and_the_exponent():
    configs = _design(points=7)
    profiles = fit_isoflop(
        _runs(configs, noise=0.0, runs_per_config=1, evaluations_per_run=1), cost=COST, n_draws=20
    )
    found = profiles.profiles()["model_size__n_params__median"].to_numpy()
    truth = np.array([_true_optimum(configs, b) for b in BUDGETS])
    assert np.abs(np.log10(found / truth)).max() < 0.005
    alpha, beta = TRUTH["alpha"], TRUTH["beta"]
    assert profiles.exponent()["median"] == pytest.approx(beta / (alpha + beta), abs=0.01)


def test_intervals_cover_the_optimum_beyond_the_largest_budget():
    """Calibration of the whole two-stage estimator, at the largest budget and a hundred times past it."""
    configs = _design()
    hits = np.zeros(2)
    n_replications = 25
    for seed in range(n_replications):
        profiles = fit_isoflop(_runs(configs, seed=seed, noise=0.005), cost=COST, n_draws=150, seed=seed)
        predicted = profiles.predict([BUDGETS[-1], 100 * BUDGETS[-1]])
        for i, budget in enumerate((BUDGETS[-1], 100 * BUDGETS[-1])):
            truth = _true_optimum(configs, budget)
            row = predicted.row(i, named=True)
            hits[i] += row["model_size__n_params__q025"] <= truth <= row["model_size__n_params__q975"]
    assert np.all(hits / n_replications >= 0.84), hits / n_replications


def test_budgets_come_from_recorded_compute_when_there_is_one():
    configs = _design(points=5)
    runs = _runs(configs, noise=0.005).with_columns(
        (6.0 * pl.col(PREDICTORS[0]) * pl.col(PREDICTORS[1]) * 1.01).alias("compute__flops")
    )
    profiles = fit_isoflop(runs, n_draws=50)
    assert profiles.budgets == pytest.approx(np.array(BUDGETS) * 1.01, rel=1e-6)
    assert profiles.cost is None
    assert set(profiles.predict([1e20]).columns) >= {"model_size__n_params__median"}


def _optimum_moved(decades):
    """The truth with its optimum moved by this many decades of model size, relative to the design.

    Moving the design instead would move the truth with it: ``simulate_runs`` states the truth relative
    to the design's own centre.
    """
    return {**TRUTH, "A": TRUTH["A"] * 10 ** (decades * (TRUTH["alpha"] + TRUTH["beta"]))}


def test_a_sweep_that_misses_the_minimum_says_so():
    """The optimum a decade above every size swept: the valley is never seen to turn up."""
    profiles = fit_isoflop(
        _runs(_design(points=5, half_width=0.3), truth=_optimum_moved(1.0)), cost=COST, n_draws=50
    )
    codes = {note.code for note in profiles.notes}
    assert codes & {"isoflop_minimum_not_bracketed", "isoflop_profile_not_convex"}


def test_too_few_sizes_or_budgets_are_errors():
    configs = _design(budgets=BUDGETS[:2], points=5) + _design(budgets=BUDGETS[2:3], points=3)
    profiles = fit_isoflop(_runs(configs), cost=COST, n_draws=50)
    codes = {note.code: note.severity for note in profiles.notes}
    assert codes["isoflop_too_few_profile_points"] == "error"
    assert codes["isoflop_too_few_budgets"] == "warning"
    with pytest.raises(IsoflopError, match="no profile can be fit"):
        fit_isoflop(_runs(_design(points=3)), cost=COST, n_draws=20)


def test_a_metric_that_improves_with_scale_has_its_maximum_found():
    metric = {"E": 2.0, "A": -0.5, "alpha": 0.3, "B": -0.375, "beta": 0.4}
    configs = _design(points=7)
    runs = simulate_runs(
        {"test_loss__ce": TRUTH, "test_metric__auroc": metric},
        configurations=configs,
        predictors=PREDICTORS,
        runs_per_config=1,
        evaluations_per_run=1,
    )
    loss = fit_isoflop(runs, cost=COST, n_draws=20).profiles()["model_size__n_params__median"]
    auroc = fit_isoflop(runs, cost=COST, target="test_metric__auroc", n_draws=20).profiles()
    assert auroc["model_size__n_params__median"].to_numpy() == pytest.approx(loss.to_numpy(), rel=0.01)


def test_profiles_and_a_correct_law_agree():
    configs = _design()
    runs = _runs(configs, noise=0.005)
    profiles = fit_isoflop(runs, cost=COST, n_draws=150)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = profiles.compare(fit(runs, n_draws=150, seed=0), [1e17, 1e19])
    assert table["agree"].all()


def test_frontiers_that_disagree_are_reported():
    """A law fit to an experiment whose exponents are swapped puts the optimum on the other side."""
    configs = _design()
    profiles = fit_isoflop(_runs(configs, noise=0.005), cost=COST, n_draws=150)
    swapped = {**TRUTH, "alpha": TRUTH["beta"], "beta": TRUTH["alpha"]}
    law = fit(_runs(configs, noise=0.005, truth=swapped), n_draws=150, seed=0)
    with pytest.warns(UserWarning, match="isoflop_law_disagreement"):
        table = profiles.compare(law, [1e21])
    assert not table["agree"].any()
