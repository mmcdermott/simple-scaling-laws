"""Empirical calibration of the uncertainty intervals.

The package's whole reason to exist is that an automated platform can act on its intervals, so the
intervals have to mean what they say. These tests generate many experiments from a known law and
count how often the nominal 95% intervals actually contain it. They are the tests that would catch a
regression in the bootstrap, which no amount of unit testing of the pieces would.

They are deliberately small -- a few dozen replications -- so they run in seconds. That is enough
resolution to catch the failure mode that matters: intervals that are systematically too narrow.
Before the small-sample calibration in
:func:`simple_scaling_laws.uncertainty.cluster_correction` was added, this design measured 74%.
"""

import itertools
import warnings

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.simulate import simulate_runs
from simple_scaling_laws.uncertainty import cluster_correction

TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}

#: Replications per design. Small enough to run in seconds, large enough that a drop to the
#: uncorrected 74% would fail: with 30 replications the standard error of a coverage estimate near
#: 95% is about 4 points.
N_REPLICATIONS = 30
N_DRAWS = 120

#: The floor asserted below. Well under the nominal 95% so ordinary Monte Carlo noise cannot fail
#: the suite, and well above the 74% the uncorrected bootstrap produced.
MINIMUM_COVERAGE = 0.82


def _coverage(runs_per_config, model_sizes, dataset_sizes, point):
    """Fraction of replications whose intervals covered the truth, for parameters and a prediction."""
    parameter_hits = dict.fromkeys(TRUTH, 0)
    prediction_hits = 0
    for seed in range(N_REPLICATIONS):
        frame = simulate_runs(
            {"test_loss__ce": TRUTH},
            model_sizes=model_sizes,
            dataset_sizes=dataset_sizes,
            runs_per_config=runs_per_config,
            evaluations_per_run=4,
            run_sd=0.02,
            eval_sd=0.03,
            seed=seed,
        )
        model = fit(frame, n_draws=N_DRAWS, seed=seed)
        intervals = model.conf_int("test_loss__ce")
        for name, value in TRUTH.items():
            parameter_hits[name] += intervals[name][0] <= value <= intervals[name][1]

        reference = np.array(
            [model.observed_domain[p]["reference"] for p in model.fitted_predictors], dtype=float
        )
        log_x = np.log(np.array([point], dtype=float)) - np.log(reference)
        truth_vector = np.array([TRUTH[name] for name in model.law.param_names])
        true_value = float(model.law.evaluate(truth_vector, log_x)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = model.predict(
                {"model_size__n_params": [point[0]], "dataset_size__n_subjects": [point[1]]}
            )
        low = float(predictions["test_loss__ce__q025"][0])
        high = float(predictions["test_loss__ce__q975"][0])
        prediction_hits += low <= true_value <= high
    return (
        {name: hits / N_REPLICATIONS for name, hits in parameter_hits.items()},
        prediction_hits / N_REPLICATIONS,
    )


def test_intervals_are_calibrated_on_a_three_by_three_design():
    """Nine configurations with two runs each: the package's bread-and-butter experiment."""
    parameters, prediction = _coverage(
        runs_per_config=2,
        model_sizes=(1e6, 1e7, 1e8),
        dataset_sizes=(1e3, 1e4, 1e5),
        point=(1e9, 1e4),
    )
    assert prediction >= MINIMUM_COVERAGE, f"prediction interval coverage was {prediction:.0%}"
    for name, rate in parameters.items():
        assert rate >= MINIMUM_COVERAGE, f"{name} interval coverage was {rate:.0%}"


def test_intervals_are_calibrated_with_a_single_run_per_configuration():
    """The harder case: no replicate runs, so run variance is inferred from the residuals."""
    parameters, prediction = _coverage(
        runs_per_config=1,
        model_sizes=(1e6, 1e7, 1e8),
        dataset_sizes=(1e3, 1e4, 1e5),
        point=(1e9, 1e4),
    )
    assert prediction >= MINIMUM_COVERAGE, f"prediction interval coverage was {prediction:.0%}"
    for name, rate in parameters.items():
        assert rate >= MINIMUM_COVERAGE, f"{name} interval coverage was {rate:.0%}"


def test_a_two_level_predictor_is_always_flagged_rather_than_silently_wrong():
    """A design that cannot identify the law must say so every time, not merely widen.

    With only two distinct dataset sizes the exponent, amplitude and offset trade off exactly, so the fitted
    parameters are arbitrary and no amount of interval widening makes them right. The package cannot fix this;
    what it must do is refuse to be quiet about it.
    """
    flagged = 0
    for seed in range(10):
        frame = simulate_runs(
            {"test_loss__ce": TRUTH},
            model_sizes=(1e6, 1e7, 1e8),
            dataset_sizes=(1e3, 1e5),
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.02,
            eval_sd=0.03,
            seed=seed,
        )
        model = fit(frame, n_draws=50, seed=seed)
        codes = {note.code for note in model.warnings if note.severity == "error"}
        flagged += "too_few_predictor_levels" in codes
    assert flagged == 10


def test_the_calibration_factor_grows_as_the_design_thins():
    """The correction must be monotone in how little the design has left over after fitting."""
    factors = [cluster_correction(c, 5) for c in (6, 9, 16, 25, 100)]
    assert all(a > b for a, b in itertools.pairwise(factors))
    assert factors[-1] == pytest.approx(1.0, abs=0.05)
    assert all(f >= 1.0 for f in factors)
