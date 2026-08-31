"""Behavior on degenerate, under-powered and misspecified experiments.

The package's contract is that it still returns a fit whenever one is mathematically possible, and records
everything it is unhappy about. These tests walk the realistic ways a small scaling experiment goes wrong and
check that each is both survivable and reported.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.simulate import simulate_runs

TRUE_LOSS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}


def codes(model) -> set[str]:
    """The warning codes a fitted model carries."""
    return {note.code for note in model.warnings}


def test_two_by_two_design_is_fit_but_flagged():
    """Four configurations cannot determine five parameters, and the artifact says so."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e6, 1e8),
        dataset_sizes=(1e3, 1e5),
        runs_per_config=1,
        evaluations_per_run=20,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    assert codes(model) >= {"underdetermined", "too_few_configurations", "single_run_per_configuration"}
    assert np.isfinite(list(model.params("test_loss__ce").values())).all()
    assert {note.severity for note in model.warnings} & {"error"}


def test_collinear_predictors_are_flagged():
    """A diagonal design cannot separate the two exponents."""
    sizes = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    rows = []
    for i, size in enumerate(sizes):
        for replicate in range(2):
            for evaluation in range(5):
                rows.append(
                    {
                        "training_run_id": f"r{i}_{replicate}",
                        "test_set_id": f"b{evaluation}",
                        "model_size__n": size,
                        "dataset_size__d": size / 100.0,
                        "test_loss__ce": 3.0 - 0.15 * i + 0.01 * replicate + 0.005 * evaluation,
                    }
                )
    model = fit(pl.DataFrame(rows), n_draws=100, seed=0)
    assert "collinear_predictors" in codes(model)


def test_no_repeated_evaluations_is_flagged():
    """Without repeated evaluations there is no separate estimate of evaluation noise."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=3,
        evaluations_per_run=1,
        run_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    assert "no_repeated_evaluations" in codes(model)
    assert model.fits["test_loss__ce"].variance_components["eval_var"] == 0.0


def test_zero_residual_degrees_of_freedom_falls_back_to_a_parametric_bootstrap():
    """With as many parameters as observations there are no residuals left to resample."""
    # Exactly five configurations for a five-parameter law: an L-shaped design so that both
    # predictors genuinely vary and neither term can be dropped.
    scales = [(1e6, 1e4), (1e7, 1e4), (1e8, 1e4), (1e7, 1e3), (1e7, 1e5)]
    rows = []
    for i, (model_size, dataset_size) in enumerate(scales):
        for evaluation in range(4):
            rows.append(
                {
                    "training_run_id": f"r{i}",
                    "test_set_id": f"b{evaluation}",
                    "model_size__n": model_size,
                    "dataset_size__d": dataset_size,
                    "test_loss__ce": 2.0 - 0.1 * i + 0.01 * evaluation,
                }
            )
    model = fit(pl.DataFrame(rows), n_draws=100, seed=0)
    assert model.fits["test_loss__ce"].goodness_of_fit["residual_dof"] == 0
    assert model.fits["test_loss__ce"].uncertainty_method == "parametric"
    assert "parametric_uncertainty" in codes(model)


def test_few_bootstrap_clusters_is_flagged():
    """Resampling over a handful of configurations gives unreliable intervals, and says so."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e6, 1e7, 1e8),
        dataset_sizes=(1e3, 1e4),
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    assert "few_bootstrap_clusters" in codes(model)


def test_a_single_configuration_yields_a_constant_fit():
    """One scale is no scaling experiment; the answer is the observed level, loudly flagged."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e7,),
        dataset_sizes=(1e4,),
        runs_per_config=4,
        evaluations_per_run=5,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=50, seed=0)
    assert model.fits["test_loss__ce"].uncertainty_method == "constant"
    assert "constant_target" in codes(model)
    predictions = model.predict({"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]})
    assert predictions["test_loss__ce__median"][0] == pytest.approx(model.params("test_loss__ce")["E"])


def test_a_saturated_metric_is_reported_as_constant(loss_frame):
    """A metric pinned at its ceiling has no law to fit, and must not stall the fitter."""
    frame = loss_frame.with_columns(test_metric__auroc=pl.lit(1.0))
    model = fit(frame, n_draws=100, seed=0)
    assert model.fits["test_metric__auroc"].uncertainty_method == "constant"
    assert model.params("test_metric__auroc") == {"E": 1.0, "A": 0.0, "alpha": 0.0, "B": 0.0, "beta": 0.0}
    assert "constant_target" in codes(model)
    # The other target is unaffected.
    assert model.params("test_loss__cross_entropy")["alpha"] == pytest.approx(0.3, abs=0.1)


def test_lack_of_fit_is_detected():
    """A target that is not a power law at all shows up as scatter far exceeding run noise."""
    rows = []
    for i, size in enumerate([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]):
        # A sawtooth in log-scale: no separable power law can follow it.
        level = 2.0 + 0.5 * (-1) ** i
        for replicate in range(3):
            for evaluation in range(4):
                rows.append(
                    {
                        "training_run_id": f"r{i}_{replicate}",
                        "test_set_id": f"b{evaluation}",
                        "model_size__n": size,
                        "dataset_size__d": 10.0 ** (3 + (i % 3)),
                        "test_loss__ce": level + 0.001 * replicate + 0.0005 * evaluation,
                    }
                )
    model = fit(pl.DataFrame(rows), n_draws=100, seed=0)
    assert "lack_of_fit" in codes(model)
    residual = model.fits["test_loss__ce"].variance_components
    assert residual["run_var_residual"] > residual["run_var_replicate"]


def test_weakly_identified_exponents_are_flagged():
    """When the data barely constrain an exponent, its interval covers most of its allowed range."""
    rows = []
    rng = np.random.default_rng(0)
    for i, size in enumerate([1e6, 2e6, 3e6, 4e6, 5e6, 6e6]):
        for replicate in range(2):
            for evaluation in range(4):
                rows.append(
                    {
                        "training_run_id": f"r{i}_{replicate}",
                        "test_set_id": f"b{evaluation}",
                        "model_size__n": size,
                        "dataset_size__d": 10.0 ** (3 + i % 2),
                        "test_loss__ce": 2.0 + rng.normal(0, 0.3),
                    }
                )
    model = fit(pl.DataFrame(rows), n_draws=200, seed=0)
    assert "weakly_identified_exponent" in codes(model)


def test_warnings_survive_into_the_artifact(tmp_path):
    """Every warning raised at fit time is readable from the saved artifact."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e6, 1e8),
        dataset_sizes=(1e3, 1e5),
        runs_per_config=1,
        evaluations_per_run=3,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=50, seed=0)
    from simple_scaling_laws import ScalingLawModel

    loaded = ScalingLawModel.load(model.save(tmp_path / "warned.slaw"))
    assert codes(loaded) == codes(model)
    assert all(note.message for note in loaded.warnings)
    assert all(isinstance(note.details, dict) for note in loaded.warnings)


def test_a_clean_experiment_raises_no_warnings():
    """A well-powered design should be quiet, or the warnings mean nothing."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=3,
        evaluations_per_run=6,
        run_sd=0.01,
        eval_sd=0.02,
        seed=0,
    )
    model = fit(frame, n_draws=200, seed=0)
    assert codes(model) == set(), [str(note) for note in model.warnings]


def test_a_single_configuration_still_estimates_run_variance():
    """One scale gives no curve, but it does say how much one new model would vary."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e7,),
        dataset_sizes=(1e4,),
        runs_per_config=25,
        evaluations_per_run=6,
        run_sd=0.10,
        eval_sd=0.02,
        seed=0,
    )
    model = fit(frame, n_draws=300, seed=0)
    components = model.fits["test_loss__ce"].variance_components
    assert np.sqrt(components["run_var"]) == pytest.approx(0.10, rel=0.5)

    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    mean = model.predict(points)
    new_run = model.predict(points, kind="new-run")

    def width(frame):
        return float(frame["test_loss__ce__q975"][0] - frame["test_loss__ce__q025"][0])

    # The level itself is uncertain (it is a mean of noisy runs), and a single new model is
    # more uncertain still.
    assert width(mean) > 0
    assert width(new_run) > width(mean)
