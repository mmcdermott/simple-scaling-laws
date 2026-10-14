"""Experiment design: the rotated isoFLOP grid, design scoring and run suggestion."""

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.cost import cost_model
from simple_scaling_laws.design import design_correlation, isoflop_grid, score_design, suggest_runs
from simple_scaling_laws.isoflop import fit_isoflop
from simple_scaling_laws.model import PredictionError
from simple_scaling_laws.simulate import simulate_runs

COST = cost_model("power", k=6.0, a=1.0, b=1.0)
PREDICTORS = ("model_size__n_params", "dataset_size__n_tokens")
TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}


def _simulate(plan, truth=TRUTH, law="separable-power", seed=0):
    configurations = [
        (n, d)
        for n, d, runs in zip(plan[PREDICTORS[0]], plan[PREDICTORS[1]], plan["runs"], strict=True)
        for _ in range(runs)
    ]
    return simulate_runs(
        {"test_loss__ce": truth},
        law=law,
        configurations=configurations,
        predictors=PREDICTORS,
        runs_per_config=1,
        evaluations_per_run=4,
        run_sd=0.02,
        eval_sd=0.03,
        seed=seed,
    )


def test_a_grid_spends_each_budget_exactly_and_centres_on_the_ratio():
    plan = isoflop_grid([1e15, 1e17, 1e19], COST, centre=20.0, points_per_budget=7)
    assert plan["cost"].to_numpy() == pytest.approx(plan["budget"].to_numpy(), rel=1e-9)
    centres = plan.filter(plan["runs"] == 2)
    ratio = centres[PREDICTORS[1]] / centres[PREDICTORS[0]]
    assert ratio.to_numpy() == pytest.approx(np.full(3, 20.0), rel=1e-6)
    assert plan.group_by("budget").len()["len"].to_list() == [7, 7, 7]


def test_a_grid_can_be_centred_on_a_fitted_frontier():
    pilot = isoflop_grid([1e15, 1e16, 1e17], COST, centre=20.0)
    runs = _simulate(pilot)
    for centre in (fit(runs, n_draws=60, seed=0), fit_isoflop(runs, cost=COST, n_draws=60)):
        plan = isoflop_grid([1e18, 1e19], COST, centre=centre, points_per_budget=5)
        middle = plan.filter(plan["runs"] == 2)[PREDICTORS[0]].to_numpy()
        if hasattr(centre, "optimal_allocation"):
            expected = centre.optimal_allocation([1e18, 1e19], COST)[f"{PREDICTORS[0]}__median"].to_numpy()
        else:
            expected = centre.predict([1e18, 1e19])[f"{PREDICTORS[0]}__median"].to_numpy()
        assert middle == pytest.approx(expected, rel=1e-9)


def test_narrow_sweeps_over_many_budgets_are_nearly_collinear():
    wide = isoflop_grid([1e15, 1e16, 1e17, 1e18, 1e19], COST, centre=20.0)
    narrow = isoflop_grid([1e15, 1e16, 1e17, 1e18, 1e19], COST, centre=20.0, half_width=(0.1, 0.1))
    assert abs(design_correlation(wide)) < 0.5
    assert design_correlation(narrow) > 0.95


def test_invalid_grids_are_refused():
    with pytest.raises(PredictionError, match="positive"):
        isoflop_grid([1e15, -1.0], COST, centre=20.0)
    with pytest.raises(PredictionError, match="one centre per budget"):
        isoflop_grid([1e15, 1e16], COST, centre=[1e6])


def test_suggestions_lie_on_the_candidate_curves_and_keep_reducing_uncertainty():
    pilot = isoflop_grid([1e15, 1e16], COST, centre=20.0, half_width=(1.2, 1.0))
    model = fit(_simulate(pilot), n_draws=200, seed=0)
    plan = suggest_runs(model, [1e20], COST, n=6, candidate_budgets=[1e17, 1e18], evaluations_per_run=4)
    assert set(plan["budget"]) <= {1e17, 1e18}
    assert plan["cost"].to_numpy() == pytest.approx(plan["budget"].to_numpy(), rel=1e-9)
    remaining = plan["remaining_variance"].to_numpy()
    assert np.all(np.diff(remaining) <= 1e-12)
    assert remaining[-1] < remaining[0] < 1.0


def test_suggestions_need_a_law_that_determines_the_optimum():
    """The multiplicative law has no interior optimum, so there is no uncertainty about one to reduce."""
    pilot = isoflop_grid([1e15, 1e16, 1e17], COST, centre=20.0)
    model = fit(_simulate(pilot), law="multiplicative-power", n_draws=60, seed=0)
    with pytest.raises(PredictionError, match="interior optimum"):
        suggest_runs(model, [1e20], COST)


def test_scoring_reports_both_frontiers_and_when_profiles_cannot_be_fit():
    """Three sizes per budget is too few for a cubic profile, so every isoFLOP fit fails and says so."""
    sparse = isoflop_grid([1e15, 1e16, 1e17], COST, centre=20.0, points_per_budget=3)
    scores = score_design(
        sparse, [TRUTH], reference=(1e8, 1e9), cost=COST, target_budgets=[1e19], replications=4, n_draws=60
    )
    by_method = {row["method"]: row for row in scores.iter_rows(named=True)}
    assert by_method["law"]["p_failed"] == 0.0
    assert by_method["law"]["rmse"] < 0.1
    assert by_method["isoflop"]["p_failed"] == 1.0


def test_a_cap_per_point_spreads_the_batch():
    pilot = isoflop_grid([1e15, 1e16], COST, centre=20.0, half_width=(1.2, 1.0))
    model = fit(_simulate(pilot), n_draws=200, seed=0)
    plan = suggest_runs(model, [1e20], COST, n=4, candidate_budgets=[1e17, 1e18], max_per_point=1)
    points = set(plan.select("budget", PREDICTORS[0]).rows())
    assert len(points) == 4
