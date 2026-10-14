"""Focused tests for the Skaling joint law (Videau, Youbi-Idrissi, Lopez-Paz & Ahuja, 2026)."""

import warnings

import numpy as np
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.laws import LawError, build_law
from simple_scaling_laws.simulate import simulate_runs

PREDICTORS = (["model_size__n"], ["dataset_size__d"])
#: E, A, alpha, beta, k, R.
PARAMS = np.array([1.0, 2.0, 0.5, 0.4, 0.6, 1.5])
POINTS = np.log(np.array([[0.1, 0.2], [1.0, 3.0], [10.0, 0.5], [30.0, 40.0]]))


def _skaling():
    return build_law("skaling-joint-power", *PREDICTORS)


def test_the_law_is_the_papers_form_with_the_amplitude_pulled_out():
    """``(A' N**-alpha + B' D**-beta)**k + E`` with ``A = A'**k`` and ``R = B' / A'``."""
    a_paper, b_paper, alpha, beta, k, offset = 3.0, 4.5, 0.5, 0.4, 0.6, 1.0
    params = np.array([offset, a_paper**k, alpha, beta, k, b_paper / a_paper])
    n, d = np.exp(POINTS[:, 0]), np.exp(POINTS[:, 1])
    expected = (a_paper * n**-alpha + b_paper * d**-beta) ** k + offset
    assert _skaling().evaluate(params, POINTS) == pytest.approx(expected, rel=1e-12)


def test_at_unit_coupling_it_is_the_separable_law():
    separable = build_law("separable-power", *PREDICTORS)
    e, a, alpha, beta, _, r = PARAMS
    coupled = _skaling().evaluate(np.array([e, a, alpha, beta, 1.0, r]), POINTS)
    assert coupled == pytest.approx(
        separable.evaluate(np.array([e, a, a * r, alpha, beta]), POINTS), rel=1e-12
    )


def test_with_a_unit_inner_data_exponent_it_is_the_kaplan_law():
    kaplan = build_law("kaplan-joint-power", *PREDICTORS)
    e, a, alpha, _, k, r = PARAMS
    coupled = _skaling().evaluate(np.array([e, a, alpha, 1.0, k, r]), POINTS)
    assert coupled == pytest.approx(kaplan.evaluate(np.array([e, a, alpha, k, r]), POINTS), rel=1e-12)


@pytest.mark.parametrize(("k", "sign"), [(0.5, -1.0), (1.0, 0.0), (1.5, 1.0)])
def test_the_coupling_exponent_sets_the_sign_of_the_interaction(k, sign):
    """The mixed difference in ``(log N, log D)`` has the sign of ``k * (k - 1)``; the loss stays monotone."""
    law, params = _skaling(), PARAMS.copy()
    params[4] = k
    h = 0.5
    corners = np.log(np.array([[1.0, 1.0], [np.e**h, 1.0], [1.0, np.e**h], [np.e**h, np.e**h]]))
    f00, f10, f01, f11 = law.evaluate(params, corners)
    mixed = f11 - f10 - f01 + f00
    assert np.sign(round(mixed, 12)) == sign
    assert f10 < f00 and f01 < f00, "more of either resource must never raise the loss"


def test_raw_scale_parameters_reproduce_the_normalized_surface():
    law = _skaling()
    reference = np.array([100.0, 16.0])
    raw = law.raw_scale_params(PARAMS, reference)
    n, d = 400.0, 64.0
    normalized = law.evaluate(PARAMS, np.log(np.array([[n / 100.0, d / 16.0]])))[0]
    by_hand = raw["E"] + raw["A"] * (n ** -raw["alpha"] + raw["R"] * d ** -raw["beta"]) ** raw["k"]
    assert by_hand == pytest.approx(normalized)


def test_it_needs_exactly_one_model_size_and_one_dataset_size_predictor():
    with pytest.raises(LawError, match="exactly one model-size and one dataset-size"):
        build_law("skaling-joint-power", ["model_size__n"], [])
    with pytest.raises(LawError, match="exactly one model-size and one dataset-size"):
        build_law("skaling-joint-power", ["model_size__n", "model_size__depth"], ["dataset_size__d"])


def test_the_coupling_exponent_is_reported_as_an_exponent():
    law = _skaling()
    assert law.n_exponents == 3
    assert law.n_nonlinear == 4


def test_on_uncoupled_data_it_finds_unit_coupling_and_raises_no_objection():
    """The separable law is an interior point of this one, so a fit to separable data identifies it.

    That is the difference from a law that merely contains the separable one as an alias: there the
    extra parameters have nothing to fit and must be flagged, here ``k = 1`` is a finding.
    """
    truth = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}
    for seed in range(4):
        frame = simulate_runs(
            {"test_loss__ce": truth},
            model_sizes=(1e6, 1e7, 1e8),
            dataset_sizes=(1e3, 1e4, 1e5),
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.02,
            eval_sd=0.03,
            seed=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit(frame, law="skaling-joint-power", n_draws=60, seed=seed)
        # The estimate's spread on this grid is about 0.03 (its 95% interval covered 93% of 60 seeds),
        # so this tolerance is several standard errors wide.
        assert model.params("test_loss__ce")["k"] == pytest.approx(1.0, abs=0.15), seed
        codes = {note.code for note in model.warnings}
        assert not codes & {"ill_conditioned_fit", "parameter_at_bound", "weakly_identified_exponent"}, seed
