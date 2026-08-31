"""Targets confined to [0, 1], which are fit on their logit.

Fitting a bounded metric on its raw scale lets the fitted asymptote leave the metric's range -- an AUROC whose
97.5th percentile is 4.9 is not a statement about a classifier. Fitting on the logit makes the bound hold by
construction, and makes the constant-variance assumption far more plausible besides, since an AUROC near 0.99
has much less room to move than one near 0.7.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import ScalingLawModel, fit
from simple_scaling_laws.data import DataError, build_dataset
from simple_scaling_laws.metrics import (
    HIGHER,
    LOWER,
    REAL,
    UNIT,
    MetricKind,
    describe,
    from_fitting_scale,
    to_fitting_scale,
)
from simple_scaling_laws.schema import discover_schema
from simple_scaling_laws.simulate import simulate_runs

AUROC_TRUTH = {"E": 2.5, "A": -1.2, "alpha": 0.35, "B": -0.8, "beta": 0.3}


def _auroc_frame(seed=0, **kwargs):
    """An AUROC experiment.

    ``simulate_runs`` generates it on the logit scale, as the model does.
    """
    return simulate_runs({"test_metric__auroc": AUROC_TRUTH}, run_sd=0.05, eval_sd=0.1, seed=seed, **kwargs)


def test_known_metrics_are_recognized_without_configuration():
    """The metrics the lab actually reports are described out of the box."""
    assert describe("auroc", is_loss=False) == MetricKind(UNIT, HIGHER, "registry")
    assert describe("auprc", is_loss=False) == MetricKind(UNIT, HIGHER, "registry")
    assert describe("brier", is_loss=False) == MetricKind(UNIT, LOWER, "registry")
    assert describe("ece", is_loss=False) == MetricKind(UNIT, LOWER, "registry")
    assert describe("cross_entropy", is_loss=True) == MetricKind(REAL, LOWER, "registry")


def test_an_unrecognized_metric_is_left_unbounded_and_says_the_direction_was_assumed():
    """Guessing a bound would be worse than not transforming at all."""
    kind = describe("some_bespoke_score", is_loss=False)
    assert kind.support == REAL
    assert kind.known_direction is False


def test_a_target_can_be_described_explicitly():
    """Anything the table does not know can be declared."""
    schema = discover_schema(
        ["training_run_id", "model_size__n", "dataset_size__d", "test_metric__weird"],
        targets={"test_metric__weird": {"support": "unit", "direction": "lower"}},
    )
    kind = schema.target("test_metric__weird").kind
    assert kind == MetricKind(UNIT, LOWER, "override")


def test_describing_an_unknown_target_is_an_error():
    """A typo in a target description should not be silently ignored."""
    with pytest.raises(Exception, match="not target columns"):
        discover_schema(
            ["training_run_id", "model_size__n", "test_loss__ce"],
            targets={"test_metric__nope": {"support": "unit"}},
        )


def test_the_logit_transform_round_trips():
    """The transform and its inverse must agree to numerical precision."""
    kind = MetricKind(UNIT, HIGHER, "registry")
    values = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    assert np.allclose(from_fitting_scale(to_fitting_scale(values, kind), kind), values)


def test_values_outside_the_unit_interval_are_rejected():
    """A target declared bounded but holding out-of-range values is a data error, not a warning."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r2", "r3"],
            "model_size__n": [1e6, 1e7, 1e8],
            "dataset_size__d": [1e4, 1e4, 1e4],
            "test_metric__auroc": [0.8, 0.9, 1.7],
        }
    )
    with pytest.raises(DataError, match=r"\[0, 1\]"):
        build_dataset(frame)


def test_saturated_values_are_nudged_and_flagged():
    """An AUROC of exactly 1.0 has no finite logit; it is moved inside and reported."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "test_set_id": ["b1", "b2"] * 3,
            "model_size__n": [1e6, 1e6, 1e7, 1e7, 1e8, 1e8],
            "dataset_size__d": [1e4] * 6,
            "test_metric__auroc": [0.80, 0.82, 0.91, 0.93, 1.0, 1.0],
        }
    )
    dataset = build_dataset(frame)
    codes = {note.code for note in dataset.notes}
    assert "saturated_values" in codes
    assert np.all(np.isfinite(dataset.observations["test_metric__auroc"].mean))


def test_predictions_stay_inside_the_metric_range_at_every_scale():
    """The bound is respected by construction, not by clipping after the fact."""
    model = fit(_auroc_frame(runs_per_config=2, evaluations_per_run=4), n_draws=300, seed=0)
    reference = np.array(
        [model.observed_domain[p]["reference"] for p in model.fitted_predictors], dtype=float
    )
    grid = np.array([[10.0**m, 10.0**d] for m in (2, 6, 10, 20) for d in (1, 4, 12)])
    draws = model.target_draws("test_metric__auroc", np.log(grid) - np.log(reference))
    assert draws.min() >= 0.0
    assert draws.max() <= 1.0


def test_the_asymptote_interval_is_a_meaningful_auroc():
    """The failure reported in issue #3: an asymptote whose upper bound was 4.9."""
    model = fit(_auroc_frame(runs_per_config=2, evaluations_per_run=6), n_draws=400, seed=0)
    huge = {"model_size__n_params": [1e18], "dataset_size__n_subjects": [1e18]}
    with pytest.warns(UserWarning):
        asymptote = model.predict(huge, quantiles=[0.025, 0.5, 0.975])
    low = float(asymptote["test_metric__auroc__q025"][0])
    high = float(asymptote["test_metric__auroc__q975"][0])
    assert 0.0 <= low <= high <= 1.0


def test_a_bounded_metric_recovers_its_generating_parameters():
    """The parameters a fit reports are the ones the data were generated from, on the logit scale."""
    model = fit(_auroc_frame(runs_per_config=2, evaluations_per_run=6), n_draws=200, seed=0)
    fitted = model.params("test_metric__auroc")
    for name, value in AUROC_TRUTH.items():
        assert fitted[name] == pytest.approx(value, abs=0.25), name


def test_a_bounded_metric_still_predicts_the_observed_values():
    """Bounding must not cost accuracy on the quantity anyone actually predicts."""
    frame = _auroc_frame(runs_per_config=2, evaluations_per_run=6)
    model = fit(frame, n_draws=200, seed=0)
    truth = (
        frame.group_by("dataset_size__n_subjects", "model_size__n_params")
        .agg(pl.col("test_metric__auroc").mean())
        .sort("model_size__n_params", "dataset_size__n_subjects")
    )
    predictions = model.predict(
        {
            "model_size__n_params": truth["model_size__n_params"].to_list(),
            "dataset_size__n_subjects": truth["dataset_size__n_subjects"].to_list(),
        }
    )
    assert np.allclose(
        predictions["test_metric__auroc__median"].to_numpy(),
        truth["test_metric__auroc"].to_numpy(),
        atol=0.02,
    )


def test_a_lower_is_better_bounded_metric_is_also_transformed():
    """Brier score is in [0, 1] too, but improves downward."""
    frame = _auroc_frame(runs_per_config=2, evaluations_per_run=4).rename(
        {"test_metric__auroc": "test_metric__brier"}
    )
    model = fit(frame, n_draws=100, seed=0)
    kind = model.target_kind("test_metric__brier")
    assert kind.support == UNIT
    assert kind.direction == LOWER


def test_an_unbounded_loss_is_not_transformed():
    """Cross-entropy has no ceiling, so nothing is gained by transforming it."""
    frame = simulate_runs(
        {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}},
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.01,
        eval_sd=0.02,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    assert model.target_kind("test_loss__cross_entropy").support == REAL
    assert model.params("test_loss__cross_entropy")["E"] == pytest.approx(1.0, abs=0.3)


def test_the_metric_kind_survives_serialization(tmp_path):
    """A reloaded artifact must back-transform exactly as the original did."""
    model = fit(_auroc_frame(runs_per_config=2, evaluations_per_run=4), n_draws=100, seed=0)
    loaded = ScalingLawModel.load(model.save(tmp_path / "auroc.slaw"))
    assert loaded.target_kind("test_metric__auroc") == model.target_kind("test_metric__auroc")
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    assert loaded.predict(points).equals(model.predict(points))


def test_run_level_aggregation_happens_on_the_fitting_scale():
    """A run's summary is the mean of its transformed rows, not the transform of their mean."""
    values = [0.60, 0.95]
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "test_set_id": ["b1", "b2"] * 3,
            "model_size__n": [1e6, 1e6, 1e7, 1e7, 1e8, 1e8],
            "dataset_size__d": [1e4] * 6,
            "test_metric__auroc": [*values, 0.7, 0.8, 0.85, 0.9],
        }
    )
    observations = build_dataset(frame).observations["test_metric__auroc"]
    kind = MetricKind(UNIT, HIGHER, "registry")
    expected = float(np.mean(to_fitting_scale(np.array(values), kind)))
    assert observations.mean[0] == pytest.approx(expected)
    assert observations.mean[0] != pytest.approx(
        float(to_fitting_scale(np.array([np.mean(values)]), kind)[0])
    )
