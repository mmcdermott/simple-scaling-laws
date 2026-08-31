"""Recognition of paired evaluation structure.

When the same bootstrap resample is scored against several trained models, the evaluation noise it contributes
is *shared*: it shifts every model's score in the same direction rather than averaging away. The package
detects that from the reused ``test_set_id`` values and reports it, and excludes the shared part from the
noise it believes comparing runs can cancel.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.data import build_dataset
from simple_scaling_laws.simulate import simulate_runs

TRUE_LOSS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9])
def test_paired_evaluation_correlation_is_recovered(fraction):
    """The estimated pairing correlation tracks the simulated shared fraction."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=30,
        run_sd=0.02,
        eval_sd=0.05,
        shared_eval_fraction=fraction,
        paired_test_sets=True,
        seed=0,
    )
    observations = build_dataset(frame).observations["test_loss__ce"]
    assert observations.eval_pair_correlation == pytest.approx(fraction, abs=0.12)
    assert observations.n_shared_pairs == 18 * 17 // 2


def test_unpaired_evaluations_report_no_pairing():
    """Without reused resample identifiers there is no paired information to find."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=10,
        eval_sd=0.05,
        paired_test_sets=False,
        seed=0,
    )
    observations = build_dataset(frame).observations["test_loss__ce"]
    assert observations.eval_pair_correlation is None
    assert observations.n_shared_pairs == 0
    assert observations.eval_var_independent == pytest.approx(observations.eval_var)


def test_shared_evaluation_noise_is_excluded_from_the_reducible_part():
    """Only the unshared part of evaluation noise averages away when runs are compared."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=30,
        run_sd=0.02,
        eval_sd=0.05,
        shared_eval_fraction=0.8,
        paired_test_sets=True,
        seed=0,
    )
    observations = build_dataset(frame).observations["test_loss__ce"]
    assert observations.eval_var_independent < observations.eval_var
    assert observations.eval_var_independent == pytest.approx(
        observations.eval_var * (1 - observations.eval_pair_correlation), rel=1e-9
    )


def test_pairing_is_reported_in_the_artifact_diagnostics():
    """A fitted model records what it found about evaluation pairing."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=20,
        run_sd=0.02,
        eval_sd=0.05,
        shared_eval_fraction=0.7,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    paired = model.diagnostics["paired_evaluation"]["test_loss__ce"]
    assert paired["eval_pair_correlation"] == pytest.approx(0.7, abs=0.15)
    assert paired["eval_sd_independent"] < paired["eval_sd"]
    assert paired["n_shared_pairs"] > 0


def test_pairing_is_detected_from_a_hand_built_table():
    """A resample that is uniformly hard for every model shows up as a positive pair correlation."""
    rows = []
    run_offsets = {"r1": 0.0, "r2": 0.1, "r3": -0.1}
    test_effects = {"b1": -0.2, "b2": 0.0, "b3": 0.2, "b4": 0.4}
    for run, run_offset in run_offsets.items():
        for test_set, effect in test_effects.items():
            rows.append(
                {
                    "training_run_id": run,
                    "test_set_id": test_set,
                    "model_size__n": 1e6,
                    "dataset_size__d": 1e4,
                    "test_loss__ce": 2.0 + run_offset + effect,
                }
            )
    observations = build_dataset(pl.DataFrame(rows)).observations["test_loss__ce"]
    assert observations.eval_pair_correlation == pytest.approx(1.0, abs=1e-9)
    assert observations.eval_var_independent == pytest.approx(0.0, abs=1e-12)


def test_partial_pairing_uses_only_shared_resamples():
    """Runs are compared only on the resamples they actually have in common."""
    rows = []
    for run in ("r1", "r2"):
        for test_set in ("b1", "b2", "b3", f"private_{run}"):
            effect = {"b1": -0.1, "b2": 0.0, "b3": 0.1}.get(test_set, 5.0)
            rows.append(
                {
                    "training_run_id": run,
                    "test_set_id": test_set,
                    "model_size__n": 1e6,
                    "dataset_size__d": 1e4,
                    "test_loss__ce": 2.0 + effect,
                }
            )
    observations = build_dataset(pl.DataFrame(rows)).observations["test_loss__ce"]
    assert observations.n_shared_pairs == 1
    assert observations.eval_pair_correlation == pytest.approx(1.0, abs=1e-9)


def test_pairing_does_not_change_the_point_estimate():
    """Detecting pairing informs the weights and diagnostics, not the curve's functional fit."""
    common = {"runs_per_config": 2, "evaluations_per_run": 20, "run_sd": 0.02, "eval_sd": 0.04, "seed": 0}
    paired = fit(
        simulate_runs({"test_loss__ce": TRUE_LOSS}, shared_eval_fraction=0.0, **common),
        n_draws=100,
        seed=0,
    )
    fitted = paired.params("test_loss__ce")
    assert np.isfinite(list(fitted.values())).all()
    assert fitted["alpha"] == pytest.approx(TRUE_LOSS["alpha"], abs=0.08)
