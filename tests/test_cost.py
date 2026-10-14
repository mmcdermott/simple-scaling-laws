"""Cost laws: declared and fitted from recorded runs, under one interface."""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws.cost import CostError, CostModel, cost_model, fit_cost_model


def _runs(cost, noise=0.0, seed=0, rows_per_run=1):
    """Recorded costs over a 4x4 grid of model and dataset sizes."""
    rng = np.random.default_rng(seed)
    grid = [(n, d) for n in (1e6, 1e7, 1e8, 1e9) for d in (1e8, 1e9, 1e10, 1e11)]
    rows = []
    for i, (n, d) in enumerate(grid):
        recorded = float(cost(n, d)) * float(np.exp(rng.normal(0.0, noise)))
        rows.extend(
            {
                "training_run_id": f"r{i}",
                "model_size__n_params": n,
                "dataset_size__n_tokens": d,
                "compute__flops": recorded,
            }
            for _ in range(rows_per_run)
        )
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    ("form", "truth"),
    [
        ("power", {"k": 6.0, "a": 1.0, "b": 1.0}),
        ("power", {"k": 2.0, "a": 1.2, "b": 0.9}),
        ("per-example", {"k": 6.0, "a": 1.0, "c": 3e8}),
    ],
)
def test_a_cost_law_is_recovered_from_noiseless_records(form, truth):
    fitted = fit_cost_model(_runs(cost_model(form, **truth)), form=form)
    for name, value in truth.items():
        assert fitted.params[name] == pytest.approx(value, rel=1e-4), name
    assert fitted.log_rmse == pytest.approx(0.0, abs=1e-6)
    assert not fitted.notes


def test_fixed_parameters_are_held_and_the_rest_fitted():
    truth = cost_model("per-example", k=6.0, a=1.0, c=3e8)
    fitted = fit_cost_model(_runs(truth, noise=0.02), form="per-example", fixed={"a": 1.0})
    assert fitted.params["a"] == 1.0
    assert fitted.fitted == ("k", "c")
    assert fitted.params["k"] == pytest.approx(6.0, rel=0.05)


def test_a_form_that_does_not_describe_the_costs_is_reported():
    """Per-example overhead makes small models dearer than any single power law can follow."""
    truth = cost_model("per-example", k=6.0, a=1.0, c=6e9)
    fitted = fit_cost_model(_runs(truth), form="power")
    assert "cost_model_poor_fit" in {note.code for note in fitted.notes}


def test_a_cost_recorded_differently_on_rows_of_one_run_is_reported():
    frame = _runs(cost_model("power", k=6.0, a=1.0, b=1.0), rows_per_run=2)
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) % 2 == 0)
        .then(pl.col("compute__flops") * 2)
        .otherwise(pl.col("compute__flops"))
    )
    fitted = fit_cost_model(frame)
    assert "cost_varies_within_run" in {note.code for note in fitted.notes}


def test_an_incomplete_declaration_says_how_to_fit_the_rest():
    with pytest.raises(CostError, match="fit_cost_model"):
        cost_model("power", k=6.0)


@pytest.mark.parametrize("params", [{"k": -1.0, "a": 1.0, "b": 1.0}, {"k": 6.0, "a": 0.0, "b": 1.0}])
def test_a_cost_that_does_not_grow_with_scale_is_refused(params):
    with pytest.raises(CostError, match="must be positive"):
        cost_model("power", **params)


def test_records_that_cannot_identify_an_exponent_are_refused():
    frame = _runs(cost_model("power", k=6.0, a=1.0, b=1.0)).filter(pl.col("model_size__n_params") == 1e6)
    with pytest.raises(CostError, match="cannot identify"):
        fit_cost_model(frame)


def test_the_iso_cost_curve_spends_the_budget_exactly():
    for cost in (cost_model("power", k=2.0, a=1.2, b=0.9), cost_model("per-example", k=6.0, a=1.0, c=3e8)):
        n = np.logspace(5, 10, 7)
        assert cost(n, cost.dataset_size_for(1e20, n)) == pytest.approx(np.full(7, 1e20), rel=1e-10)


def test_a_fitted_cost_model_round_trips():
    fitted = fit_cost_model(_runs(cost_model("power", k=6.0, a=1.0, b=1.0), noise=0.01))
    assert CostModel.from_dict(fitted.to_dict()) == fitted
