"""Focused tests for the separable interaction power law."""

import warnings

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.laws import build_law
from simple_scaling_laws.model import raw_scale_params
from simple_scaling_laws.simulate import simulate_runs

LAW = "separable-interaction-power"


def test_interaction_law_can_reverse_model_size_effect_by_dataset_size():
    """The same surface can prefer larger models at high D and smaller models at low D."""
    law = build_law(
        "separable-interaction-power",
        ["model_size__n"],
        ["dataset_size__d"],
    )
    # E + A*N^-alpha + B*D^-beta - C*N^-gamma*D^-delta.
    params = np.array([1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0])

    # Values are normalized coordinates. At low D the negative interaction fades quickly as N
    # grows, so loss rises; at high D the interaction is negligible and the ordinary N power wins.
    low_data = np.log(np.array([[1.0, 0.25], [4.0, 0.25]]))
    high_data = np.log(np.array([[1.0, 16.0], [4.0, 16.0]]))
    low_values = law.evaluate(params, low_data)
    high_values = law.evaluate(params, high_data)

    assert low_values[1] > low_values[0]
    assert high_values[1] < high_values[0]


def test_interaction_zero_reduces_to_separable_surface():
    """Setting C to zero exactly recovers the corresponding separable law."""
    interaction = build_law(
        "separable-interaction-power",
        ["model_size__n"],
        ["dataset_size__d"],
    )
    separable = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
    log_x = np.log(np.array([[0.5, 2.0], [1.0, 1.0], [4.0, 8.0]]))

    interaction_params = np.array([1.0, 2.0, 3.0, 0.0, 0.4, 0.25, 0.7, 0.6])
    separable_params = np.array([1.0, 2.0, 3.0, 0.4, 0.25])
    assert np.allclose(
        interaction.evaluate(interaction_params, log_x),
        separable.evaluate(separable_params, log_x),
    )


def test_interaction_raw_scale_conversion():
    """Main and interaction amplitudes absorb the correct normalization factors."""
    law = build_law(
        "separable-interaction-power",
        ["model_size__n"],
        ["dataset_size__d"],
    )
    params = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.75, 0.5])
    reference = np.array([100.0, 16.0])
    raw = raw_scale_params(law, params, reference)

    assert raw["A"] == 20.0
    assert raw["B"] == 6.0
    assert np.isclose(raw["C"], 4.0 * 100.0**0.75 * 16.0**0.5)
    assert raw["gamma"] == 0.75
    assert raw["delta"] == 0.5


def test_interaction_jacobian_matches_finite_difference():
    """The interaction family supplies an accurate analytic Jacobian."""
    law = build_law(
        "separable-interaction-power",
        ["model_size__n"],
        ["dataset_size__d"],
    )
    params = np.array([1.0, 2.0, 1.5, 0.8, 0.3, 0.2, 0.6, 0.4])
    log_x = np.log(np.array([[0.5, 0.25], [1.0, 2.0], [4.0, 8.0]]))
    analytic = law.jacobian(params, log_x)
    numerical = np.empty_like(analytic)
    for j in range(law.n_params):
        step = 1e-6 * max(1.0, abs(params[j]))
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (law.evaluate(up, log_x) - law.evaluate(down, log_x)) / (2 * step)
    assert np.allclose(analytic, numerical, atol=1e-6, rtol=1e-5)


def test_interaction_supports_multiple_scale_predictors():
    """The family composes one joint interaction over arbitrary scale coordinates."""
    law = build_law(
        "separable-interaction-power",
        ["model_size__params", "model_size__depth"],
        ["dataset_size__subjects"],
    )
    assert law.n_linear == 5
    assert law.n_exponents == 6
    log_x = np.zeros((2, 3))
    params = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    values = law.evaluate(params, log_x)
    assert np.allclose(values, 1.0 + 2.0 + 3.0 + 4.0 - 0.5)


# --- End to end: the law has to survive the fitter, the bootstrap and the diagnostics, not just its
# own algebra. Sixteen configurations is the smallest square grid that leaves the eight-parameter law
# any spare variation.

INTERACTION_TRUTH = {
    "E": 1.0,
    "A": 2.0,
    "B": 3.0,
    "C": 1.5,
    "alpha": 0.3,
    "beta": 0.3,
    "gamma": 0.15,
    "delta": 0.3,
}
SEPARABLE_TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}
MODEL_SIZES = (1e6, 1e7, 1e8, 1e9)
DATASET_SIZES = (1e3, 1e4, 1e5, 1e6)
#: An interior point, then one decade beyond the grid in model size, in data, and in both.
POINTS = {
    "model_size__n_params": [3e7, 1e10, 1e7, 1e10],
    "dataset_size__n_subjects": [3e4, 1e4, 1e7, 1e7],
}


def _simulate(truth, law, seed, runs_per_config=2, run_sd=0.02, eval_sd=0.03, evaluations_per_run=4):
    return simulate_runs(
        {"test_loss__ce": truth},
        model_sizes=MODEL_SIZES,
        dataset_sizes=DATASET_SIZES,
        law=law,
        runs_per_config=runs_per_config,
        evaluations_per_run=evaluations_per_run,
        run_sd=run_sd,
        eval_sd=eval_sd,
        seed=seed,
    )


def _true_values(model, truth):
    reference = np.array([model.observed_domain[p]["reference"] for p in model.fitted_predictors])
    points = np.column_stack([POINTS[p] for p in model.fitted_predictors])
    vector = np.array([truth[name] for name in model.law.param_names])
    return model.law.evaluate(vector, np.log(points) - np.log(reference))


def _predict(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(POINTS)


def test_fit_recovers_a_noiseless_interaction_surface():
    """Without noise the fitter must find the generating law, parameters included."""
    frame = _simulate(INTERACTION_TRUTH, LAW, seed=0, runs_per_config=1, run_sd=0.0, eval_sd=0.0)
    model = fit(frame, law=LAW, n_draws=20, seed=0)
    fitted = model.params("test_loss__ce")
    for name, truth in INTERACTION_TRUTH.items():
        assert fitted[name] == pytest.approx(truth, abs=1e-3), name
    assert _predict(model)["test_loss__ce__median"].to_numpy() == pytest.approx(
        _true_values(model, INTERACTION_TRUTH), abs=1e-4
    )


def test_fit_recovers_the_reversal_from_noisy_runs():
    """The reason the law exists: larger models hurt when data are scarce, and help when they are not."""
    model = fit(_simulate(INTERACTION_TRUTH, LAW, seed=0), law=LAW, n_draws=100, seed=0)
    assert not [note for note in model.warnings if note.severity == "error"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict(
            {
                "model_size__n_params": [1e7, 1e9, 1e7, 1e9],
                "dataset_size__n_subjects": [1e3, 1e3, 1e6, 1e6],
            }
        )
    scarce_small, scarce_large, ample_small, ample_large = predictions["test_loss__ce__median"]
    assert scarce_large > scarce_small
    assert ample_large < ample_small


def test_prediction_intervals_are_calibrated_for_the_interaction_law():
    """The bootstrap was calibrated on the separable law; it has to hold for this one too.

    Measured at 92-98% over forty replications when this test was written. The floor is the one
    ``tests/test_coverage.py`` uses, for the same reason: loose enough for Monte Carlo noise at this
    replication count, tight enough to catch intervals that are systematically too narrow.
    """
    n_replications = 20
    hits = np.zeros(len(POINTS["model_size__n_params"]))
    for seed in range(n_replications):
        model = fit(_simulate(INTERACTION_TRUTH, LAW, seed=seed), law=LAW, n_draws=120, seed=seed)
        predictions = _predict(model)
        truth = _true_values(model, INTERACTION_TRUTH)
        hits += (predictions["test_loss__ce__q025"].to_numpy() <= truth) & (
            truth <= predictions["test_loss__ce__q975"].to_numpy()
        )
    coverage = hits / n_replications
    assert coverage.min() >= 0.82, f"prediction interval coverage was {coverage}"


def test_data_without_an_interaction_are_flagged_and_still_predicted():
    """On separable data the joint term can only alias the separable ones.

    The curve is still right -- the law contains the separable one -- but the individual parameters are not
    identified, and the package has to say so every time rather than report an arbitrary asymptote in silence.
    """
    unidentified = {
        "ill_conditioned_fit",
        "parameter_at_bound",
        "weakly_identified_exponent",
        "optimizer_nonconvergence",
    }
    for seed in range(6):
        frame = _simulate(SEPARABLE_TRUTH, "separable-power", seed=seed)
        model = fit(frame, law=LAW, n_draws=60, seed=seed)
        assert unidentified & {note.code for note in model.warnings}, seed
        simple = fit(frame, n_draws=60, seed=seed)
        assert _predict(model)["test_loss__ce__median"].to_numpy()[0] == pytest.approx(
            _predict(simple)["test_loss__ce__median"].to_numpy()[0], abs=0.02
        )


def test_an_identified_fit_is_not_called_ill_conditioned():
    """The check must stay quiet on the designs the package is built for."""
    for law, truth in (("separable-power", SEPARABLE_TRUTH), (LAW, INTERACTION_TRUTH)):
        for seed in range(5):
            model = fit(_simulate(truth, law, seed=seed), law=law, n_draws=30, seed=seed)
            assert "ill_conditioned_fit" not in {note.code for note in model.warnings}, (law, seed)
