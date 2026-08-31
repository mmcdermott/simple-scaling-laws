"""Fitting every recognized target in one invocation.

Fitting a metric costs almost nothing once the data are reduced to run means, so one call fits the test loss,
the train loss and every alternative metric, each with its own parameters and its own uncertainty draws.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import ScalingLawModel, fit
from simple_scaling_laws.model import PredictionError
from simple_scaling_laws.simulate import simulate_runs

PARAMS = {
    "test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.30, "B": 1.5, "beta": 0.25},
    "train_loss__cross_entropy": {"E": 0.7, "A": 2.0, "alpha": 0.35, "B": 1.5, "beta": 0.30},
    "test_metric__auroc": {"E": 0.92, "A": -0.15, "alpha": 0.40, "B": -0.08, "beta": 0.30},
    "test_metric__auprc": {"E": 0.75, "A": -0.20, "alpha": 0.35, "B": -0.10, "beta": 0.25},
    "train_metric__accuracy": {"E": 0.95, "A": -0.10, "alpha": 0.45, "B": -0.05, "beta": 0.35},
}


@pytest.fixture(scope="module")
def model():
    """One fit covering a loss, a train loss, two test metrics and a train metric."""
    frame = simulate_runs(PARAMS, runs_per_config=2, evaluations_per_run=4, run_sd=0.01, eval_sd=0.02, seed=0)
    return fit(frame, n_draws=200, seed=0)


def test_every_recognized_target_is_fit(model):
    """All five targets appear, with the test loss designated primary."""
    assert set(model.targets) == set(PARAMS)
    assert model.primary_target == "test_loss__cross_entropy"
    assert model.targets[0] == model.primary_target


def test_each_target_gets_its_own_parameters(model):
    """Parameters are estimated per target, not shared."""
    for target, truth in PARAMS.items():
        fitted = model.params(target)
        assert fitted["alpha"] == pytest.approx(truth["alpha"], abs=0.12), target
        assert np.sign(fitted["A"]) == np.sign(truth["A"]), target


def test_each_target_gets_its_own_draws(model):
    """Uncertainty is per target too, and the draws genuinely differ between them."""
    for target in PARAMS:
        assert model.draws[target].params.shape == (200, 5)
    loss = model.draws["test_loss__cross_entropy"].params
    auroc = model.draws["test_metric__auroc"].params
    assert not np.allclose(loss, auroc)


def test_target_roles_are_recorded(model):
    """Losses and metrics are distinguished, which is what sets the amplitude constraint."""
    roles = {name: model.fits[name].role for name in PARAMS}
    assert roles["test_loss__cross_entropy"] == "test_loss"
    assert roles["train_metric__accuracy"] == "train_metric"
    assert model.fits["test_loss__cross_entropy"].signed_amplitude is False
    assert model.fits["test_metric__auroc"].signed_amplitude is True


def test_prediction_covers_all_targets_by_default(model):
    """A prediction request with no target list returns every fitted target."""
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    predictions = model.predict(points)
    for target in PARAMS:
        assert f"{target}__median" in predictions.columns
        assert f"{target}__q025" in predictions.columns
        assert f"{target}__q975" in predictions.columns


def test_targets_can_be_selected(model):
    """Callers can restrict prediction to the targets they care about."""
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    predictions = model.predict(points, targets=["test_metric__auroc"])
    target_columns = [c for c in predictions.columns if "__q" in c or c.endswith("__median")]
    assert all(c.startswith("test_metric__auroc") for c in target_columns)


def test_unknown_targets_are_rejected(model):
    """A typo in a target name fails loudly."""
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    with pytest.raises(PredictionError, match="No fit for target"):
        model.predict(points, targets=["test_metric__nonexistent"])


def test_all_targets_survive_serialization(tmp_path, model):
    """Every target's fit and draws are persisted and restored."""
    path = model.save(tmp_path / "many.slaw")
    loaded = ScalingLawModel.load(path)
    assert set(loaded.targets) == set(PARAMS)
    for target in PARAMS:
        assert np.array_equal(loaded.draws[target].params, model.draws[target].params)
        assert loaded.fits[target].role == model.fits[target].role


def test_a_target_with_missing_values_is_still_fit():
    """A metric recorded for only some runs is fit on the runs that have it."""
    frame = simulate_runs(
        {k: PARAMS[k] for k in ("test_loss__cross_entropy", "test_metric__auroc")},
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.01,
        eval_sd=0.02,
        seed=0,
    )
    holed = frame.with_columns(
        test_metric__auroc=pl.when(pl.col("model_size__n_params") == 1e6)
        .then(None)
        .otherwise(pl.col("test_metric__auroc"))
    )
    model = fit(holed, n_draws=100, seed=0)
    assert set(model.targets) == {"test_loss__cross_entropy", "test_metric__auroc"}
    assert model.fits["test_metric__auroc"].goodness_of_fit["n_observations"] == 12
    assert model.fits["test_loss__cross_entropy"].goodness_of_fit["n_observations"] == 18
    assert "dropped_rows" in {note.code for note in model.warnings}


def test_a_fully_missing_target_is_skipped_with_a_warning():
    """A metric column that is entirely null does not derail the rest of the fit."""
    frame = simulate_runs(
        {k: PARAMS[k] for k in ("test_loss__cross_entropy", "test_metric__auroc")},
        runs_per_config=2,
        evaluations_per_run=3,
        run_sd=0.01,
        eval_sd=0.02,
        seed=0,
    )
    empty = frame.with_columns(test_metric__auroc=pl.lit(None, dtype=pl.Float64))
    model = fit(empty, n_draws=50, seed=0)
    assert model.targets == ("test_loss__cross_entropy",)
    assert "empty_target" in {note.code for note in model.warnings}
