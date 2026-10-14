"""What every law family owes the rest of the package, checked end to end.

A law's own algebra -- its limits, its Jacobian, its raw-scale conversion -- is tested beside the law. That is
not enough to trust it. The package's value is in what happens after the algebra: the fitter has to find the
law from a small grid, the bootstrap has to produce intervals that mean what they say, and the diagnostics
have to speak up when the data cannot support the law that was asked for. Those are properties of the law
*and* the machinery together, so they are tested here, once, for every family. Adding a law means adding a row
to :data:`CASES`.
"""

import dataclasses
import itertools
import warnings

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.diagnostics import conditioning
from simple_scaling_laws.laws import LawError, available_laws, build_law
from simple_scaling_laws.simulate import simulate_runs

TARGET = "test_loss__ce"
SEPARABLE_TRUTH = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}

#: Replications behind each coverage estimate, and the draws per fit. Seeds are fixed, so the
#: estimate is a deterministic number rather than a flaky one; the floors below sit under the values
#: measured on forty replications when each law was added.
N_REPLICATIONS = 15
N_DRAWS = 120

#: Floors on mean and worst-point prediction-interval coverage, those of ``test_coverage.py``.
MEAN_COVERAGE, MIN_COVERAGE = 0.88, 0.8


@dataclasses.dataclass(frozen=True)
class Case:
    """A law, a truth to generate from, and the smallest square grid the law is meant for."""

    truth: dict[str, float]
    levels: int
    #: Laws this one reduces to exactly at some parameter value, so can be over-fit to.
    contains_separable: bool = False
    #: Whether the law is defined over model size alone, so the experiment is a sweep, not a grid.
    single_predictor: bool = False

    @property
    def model_sizes(self):
        return tuple(10.0 ** (6 + i) for i in range(self.levels))

    @property
    def dataset_sizes(self):
        return () if self.single_predictor else tuple(10.0 ** (3 + i) for i in range(self.levels))

    @property
    def predictors(self):
        both = (["model_size__n_params"], ["dataset_size__n_subjects"])
        return (both[0], []) if self.single_predictor else both

    @property
    def points(self):
        """An interior point, then a decade beyond the grid in model size, in data, and in both."""
        big_model = self.model_sizes[-1] * 10
        if self.single_predictor:
            return {"model_size__n_params": [3e7, 3e10, big_model]}
        big_data = self.dataset_sizes[-1] * 10
        return {
            "model_size__n_params": [3e7, big_model, 1e7, big_model],
            "dataset_size__n_subjects": [3e4, 1e4, big_data, big_data],
        }

    @property
    def separable_truth(self):
        """A separable law over the same predictors, for over-fitting this one to."""
        return {"E": 1.0, "A": 2.0, "alpha": 0.3} if self.single_predictor else SEPARABLE_TRUTH


CASES = {
    "separable-interaction-power": Case(
        {"E": 1.0, "A": 2.0, "B": 3.0, "C": 1.5, "alpha": 0.3, "beta": 0.3, "gamma": 0.15, "delta": 0.3},
        levels=4,
        contains_separable=True,
    ),
    "shifted-separable-power": Case(
        {"E": 1.0, "A": 2.0, "B": 1.5, "alpha": 0.5, "beta": 0.4, "s_n_params": 0.3, "s_n_subjects": 0.2},
        levels=4,
        contains_separable=True,
    ),
    "kaplan-joint-power": Case({"E": 1.0, "A": 2.0, "rho": 0.5, "beta": 0.6, "R": 1.0}, levels=4),
    # Contains the separable law at ``k = 1``, but as an interior point that the data identify rather
    # than an alias, so it is not in NESTING; see test_skaling_law.py.
    "skaling-joint-power": Case(
        {"E": 1.0, "A": 2.0, "alpha": 0.5, "beta": 0.4, "k": 0.5, "R": 1.0},
        levels=3,
    ),
    # Contains the separable law only as a limit -- an ever-higher plateau with ever-smaller weights
    # -- which is exactly what a fit to separable data runs toward.
    "bounded-rational-power": Case(
        {"E": 1.0, "I": 4.0, "A": 1.0, "B": 0.8, "alpha": 0.5, "beta": 0.4},
        levels=4,
        contains_separable=True,
    ),
    # Eight scales for six parameters on one axis: the design on which the bootstrap needed its
    # noise floor (see ``uncertainty.corrected_residuals``) to stop under-covering.
    "broken-power": Case(
        {"E": 1.0, "A": 2.0, "c0": 0.15, "c1": 0.5, "d": 1.0, "f": 1.0},
        levels=8,
        contains_separable=True,
        single_predictor=True,
    ),
}
NESTING = sorted(name for name, case in CASES.items() if case.contains_separable)


def _simulate(case, law, seed, truth=None, noise=True, runs_per_config=2):
    return simulate_runs(
        {TARGET: case.truth if truth is None else truth},
        model_sizes=case.model_sizes,
        dataset_sizes=case.dataset_sizes,
        law=law,
        runs_per_config=runs_per_config if noise else 1,
        evaluations_per_run=4 if noise else 1,
        run_sd=0.02 if noise else 0.0,
        eval_sd=0.03 if noise else 0.0,
        seed=seed,
    )


def _predict(model, case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(case.points)


def _true_values(model, case):
    reference = np.array([model.observed_domain[p]["reference"] for p in model.fitted_predictors])
    points = np.column_stack([case.points[p] for p in model.fitted_predictors])
    vector = np.array([case.truth[name] for name in model.law.param_names])
    return model.law.evaluate(vector, np.log(points) - np.log(reference))


def test_every_registered_law_beyond_the_original_two_is_covered_here():
    """A law added to the registry without a row here has not been validated."""
    assert set(CASES) == set(available_laws()) - {"separable-power", "multiplicative-power"}


@pytest.mark.parametrize("law", sorted(CASES))
def test_noiseless_data_are_recovered_exactly(law):
    """Without noise the fitter must find the generating law, parameters included."""
    case = CASES[law]
    model = fit(_simulate(case, law, seed=0, noise=False), law=law, n_draws=20, seed=0)
    fitted = model.params(TARGET)
    for name, truth in case.truth.items():
        assert fitted[name] == pytest.approx(truth, abs=1e-3), name
    assert _predict(model, case)[f"{TARGET}__median"].to_numpy() == pytest.approx(
        _true_values(model, case), abs=1e-4
    )


@pytest.mark.parametrize("law", sorted(CASES))
def test_prediction_intervals_are_calibrated(law):
    """The bootstrap was calibrated on the separable law; it has to hold for every other one too.

    Each point is a different question -- inside the grid, and a decade beyond it in each direction
    -- and none may be systematically too narrow. The floors are those of ``test_coverage.py``.
    """
    case = CASES[law]
    hits = np.zeros(len(case.points["model_size__n_params"]))
    for seed in range(N_REPLICATIONS):
        model = fit(_simulate(case, law, seed=seed), law=law, n_draws=N_DRAWS, seed=seed)
        predictions = _predict(model, case)
        truth = _true_values(model, case)
        hits += (predictions[f"{TARGET}__q025"].to_numpy() <= truth) & (
            truth <= predictions[f"{TARGET}__q975"].to_numpy()
        )
    coverage = hits / N_REPLICATIONS
    assert coverage.mean() >= MEAN_COVERAGE, f"mean coverage was {coverage.mean():.0%}"
    assert coverage.min() >= MIN_COVERAGE, f"coverage by point was {coverage}"


@pytest.mark.parametrize("law", sorted(CASES))
def test_the_recommended_design_raises_no_objection(law):
    """On the grid a law is meant for, with data it generated, the diagnostics must stay quiet."""
    case = CASES[law]
    for seed in range(4):
        model = fit(_simulate(case, law, seed=seed), law=law, n_draws=30, seed=seed)
        codes = {note.code for note in model.warnings}
        assert not {note.code for note in model.warnings if note.severity == "error"}, seed
        assert not codes & {"ill_conditioned_fit", "too_few_configurations", "parameter_at_bound"}, seed


@pytest.mark.parametrize("law", NESTING)
def test_a_richer_law_on_separable_data_is_flagged_and_still_predicts(law):
    """A law that contains the separable one can only alias it when the data are separable.

    The curve is still right, but the extra parameters have nothing of their own to fit. The package has to
    say so every time rather than report an arbitrary parameter in silence.
    """
    case = CASES[law]
    unidentified = {
        "ill_conditioned_fit",
        "parameter_at_bound",
        "weakly_identified_exponent",
        "optimizer_nonconvergence",
    }
    for seed in range(5):
        frame = _simulate(case, "separable-power", seed=seed, truth=case.separable_truth)
        rich = fit(frame, law=law, n_draws=60, seed=seed)
        simple = fit(frame, n_draws=60, seed=seed)
        assert unidentified & {note.code for note in rich.warnings}, seed
        assert _predict(rich, case)[f"{TARGET}__median"][0] == pytest.approx(
            _predict(simple, case)[f"{TARGET}__median"][0], abs=0.02
        )


def _instance(law):
    """A law over the predictors its case uses, or over one of each for the original two laws."""
    predictors = CASES[law].predictors if law in CASES else Case({}, 3).predictors
    return build_law(law, *predictors)


def _identified(instance, params, levels):
    """Whether a full grid with this many levels per predictor pins down every parameter."""
    axis = np.linspace(-np.log(30.0), np.log(30.0), levels)
    grid = np.array(list(itertools.product(axis, repeat=len(instance.predictors))))
    number, _ = conditioning(instance, params, grid, np.ones(len(grid)))
    return len(grid) >= instance.n_params and number < 1e8


@pytest.mark.parametrize("law", sorted(available_laws()))
def test_a_law_is_identified_at_its_declared_minimum_levels(law):
    """At the declared minimum number of levels the Jacobian has full column rank.

    ``too_few_predictor_levels`` rests on this number, so it is checked against the law's own derivatives
    rather than asserted.

    Where a law asks for more than the package-wide floor of three, the extra level must be necessary too --
    one fewer and some direction in parameter space leaves every fitted value unchanged.
    """
    instance = _instance(law)
    truth = CASES[law].truth if law in CASES else SEPARABLE_TRUTH
    params = np.array([truth[name] for name in instance.param_names])
    required = max(instance.min_predictor_levels)
    assert _identified(instance, params, required)
    if required > 3:
        assert not _identified(instance, params, required - 1)


@pytest.mark.parametrize("law", sorted(available_laws()))
def test_raw_scale_parameters_describe_the_same_surface(law):
    """Evaluating raw-scale parameters at raw predictor values must reproduce the normalized fit."""
    instance = _instance(law)
    truth = CASES[law].truth if law in CASES else SEPARABLE_TRUTH
    params = np.array([truth[name] for name in instance.param_names])
    k = len(instance.predictors)
    reference = np.array([3e7, 2e4])[:k]
    raw = instance.raw_scale_params(params, reference)
    raw_vector = np.array([raw[name] for name in instance.param_names])
    x = np.array([[1e6, 1e3], [4e7, 5e4], [2e9, 8e5]])[:, :k]
    assert instance.evaluate(raw_vector, np.log(x)) == pytest.approx(
        instance.evaluate(params, np.log(x / reference)), rel=1e-10
    )


def test_a_shifted_law_on_three_levels_is_an_error_not_a_quiet_fit():
    """Nine configurations exceed the law's seven parameters, yet cannot identify it."""
    case = dataclasses.replace(CASES["shifted-separable-power"], levels=3)
    model = fit(_simulate(case, "shifted-separable-power", seed=0), law="shifted-separable-power", n_draws=30)
    flagged = [note for note in model.warnings if note.code == "too_few_predictor_levels"]
    assert len(flagged) == 2
    assert all(note.severity == "error" and note.details["required"] == 4 for note in flagged)


def test_a_law_that_cannot_be_built_says_which_predictor_the_data_removed():
    """A fixed-arity law fails because of the experiment, and the message has to point there."""
    case = CASES["kaplan-joint-power"]
    frame = _simulate(case, "kaplan-joint-power", seed=0).filter(
        pl.col("dataset_size__n_subjects") == case.dataset_sizes[0]
    )
    with pytest.raises(LawError, match=r"dataset_size__n_subjects.*single value"):
        fit(frame, law="kaplan-joint-power", n_draws=20)
