"""Recovery of the two variance components, and of the difference between them.

The package's central claim is that it distinguishes *training-run* stochasticity from
*finite-evaluation-set* noise. These tests hold it to that: repeated evaluations of one trained
model must not be counted as independent evidence about the scaling law, and repeated trained models
at one scale must be.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.data import build_dataset
from simple_scaling_laws.simulate import simulate_runs

TRUE_LOSS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}


def _fit(run_sd, eval_sd, runs_per_config, evaluations_per_run, seed=0, n_draws=200):
    """Fit a synthetic experiment with the given noise structure."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=runs_per_config,
        evaluations_per_run=evaluations_per_run,
        run_sd=run_sd,
        eval_sd=eval_sd,
        paired_test_sets=False,
        seed=seed,
    )
    return fit(frame, n_draws=n_draws, seed=0)


def test_run_variance_is_recovered_from_replicates():
    """Independently trained models at the same scale reveal the training-run variance."""
    model = _fit(run_sd=0.10, eval_sd=0.02, runs_per_config=6, evaluations_per_run=6)
    components = model.fits["test_loss__ce"].variance_components
    assert components["source"] == "replicates"
    assert np.sqrt(components["run_var"]) == pytest.approx(0.10, rel=0.35)


def test_evaluation_variance_is_recovered_from_repeated_evaluations():
    """Repeated evaluations of one trained model reveal the evaluation noise."""
    model = _fit(run_sd=0.02, eval_sd=0.08, runs_per_config=2, evaluations_per_run=20)
    components = model.fits["test_loss__ce"].variance_components
    assert np.sqrt(components["eval_var"]) == pytest.approx(0.08, rel=0.15)


def test_the_two_components_are_not_confused():
    """Large evaluation noise must not be reported as large training-run noise, or vice versa."""
    loud_evaluation = _fit(run_sd=0.01, eval_sd=0.20, runs_per_config=4, evaluations_per_run=20)
    loud_training = _fit(run_sd=0.20, eval_sd=0.01, runs_per_config=4, evaluations_per_run=20)
    quiet = loud_evaluation.fits["test_loss__ce"].variance_components
    loud = loud_training.fits["test_loss__ce"].variance_components
    assert np.sqrt(quiet["run_var"]) < 0.08
    assert np.sqrt(loud["run_var"]) > 0.10
    assert np.sqrt(quiet["eval_var"]) > np.sqrt(loud["eval_var"])


def test_run_variance_falls_back_to_residuals_without_replicates():
    """Without replicate runs the estimate is residual-based, and says so."""
    model = _fit(run_sd=0.05, eval_sd=0.02, runs_per_config=1, evaluations_per_run=6)
    components = model.fits["test_loss__ce"].variance_components
    assert components["source"] == "residual"
    assert components["run_var_replicate"] is None
    codes = {note.code for note in model.warnings}
    assert "single_run_per_configuration" in codes


def test_more_evaluations_do_not_masquerade_as_more_training_evidence():
    """Multiplying evaluation resamples must not shrink parameter intervals like new runs would.

    Both fits below train exactly the same number of models; one simply scores each model on far
    more bootstrap resamples. A fitter that treated evaluation rows as independent observations
    would report intervals roughly ``sqrt(40 / 4)`` times narrower for the second fit.
    """
    full = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=40,
        run_sd=0.05,
        eval_sd=0.05,
        seed=0,
    )
    # Subsetting, rather than regenerating, guarantees both fits see the *same* trained models with
    # the *same* training-run offsets -- only the number of evaluation resamples differs.
    subset = full.filter(pl.col("test_set_id").is_in([f"boot_{i:03d}" for i in range(4)]))

    many = fit(full, n_draws=400, seed=0)
    few = fit(subset, n_draws=400, seed=0)
    few_width = np.subtract(*reversed(few.conf_int("test_loss__ce")["alpha"]))
    many_width = np.subtract(*reversed(many.conf_int("test_loss__ce")["alpha"]))

    assert few.manifest["n_training_runs"] == many.manifest["n_training_runs"] == 18
    assert many.manifest["n_evaluation_rows"] == 10 * few.manifest["n_evaluation_rows"]
    assert many_width > few_width / 1.6, "intervals shrank as if evaluations were runs"
    # The honest shrinkage is only the reduction in the noise of each run's mean, which for these
    # variance components is about 10%, nowhere near the sqrt(10) an evaluation-row fitter implies.
    assert many_width < few_width * 1.6


def test_more_training_runs_do_narrow_the_intervals():
    """Genuinely independent trained models are the evidence that sharpens a scaling law."""
    few = _fit(run_sd=0.08, eval_sd=0.02, runs_per_config=2, evaluations_per_run=4, n_draws=400)
    many = _fit(run_sd=0.08, eval_sd=0.02, runs_per_config=8, evaluations_per_run=4, n_draws=400)
    few_width = np.subtract(*reversed(few.conf_int("test_loss__ce")["alpha"]))
    many_width = np.subtract(*reversed(many.conf_int("test_loss__ce")["alpha"]))
    assert many_width < few_width


def test_run_means_are_the_unit_of_observation():
    """The dataset reduces each trained model to one mean plus a count."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=7,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    dataset = build_dataset(frame)
    observations = dataset.observations["test_loss__ce"]
    assert observations.n_runs == 18
    assert observations.n_eval.tolist() == [7] * 18
    assert observations.within_dof == 18 * 6
    grouped = frame.group_by("training_run_id").mean().sort("training_run_id")
    assert np.allclose(observations.mean, grouped["test_loss__ce"].to_numpy())


def test_run_sd_draws_are_available_for_new_run_predictions():
    """Uncertainty about the run-level scale is carried, not collapsed to a point."""
    model = _fit(run_sd=0.08, eval_sd=0.02, runs_per_config=4, evaluations_per_run=4, n_draws=300)
    draws = model.draws["test_loss__ce"]
    assert draws.run_sd.shape == (300,)
    assert np.ptp(draws.run_sd) > 0
    assert draws.run_deviations.size == 4 * 9


def test_new_run_predictions_are_wider_than_mean_predictions():
    """Predicting one new model must be less certain than predicting the expected curve."""
    model = _fit(run_sd=0.10, eval_sd=0.02, runs_per_config=4, evaluations_per_run=4, n_draws=400)
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    mean = model.predict(points, kind="mean")
    new_run = model.predict(points, kind="new-run")
    mean_width = float(mean["test_loss__ce__q975"][0] - mean["test_loss__ce__q025"][0])
    new_width = float(new_run["test_loss__ce__q975"][0] - new_run["test_loss__ce__q025"][0])
    assert new_width > mean_width
    assert new_width > 2 * model.fits["test_loss__ce"].run_sd
