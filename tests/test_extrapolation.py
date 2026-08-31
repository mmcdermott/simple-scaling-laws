"""Annotation of points outside the observed scaling domain.

Interpolation and extrapolation go through the same ``predict`` call -- the difference is reported,
not enforced -- because the whole point of fitting a scaling law is to ask about scales you have not
run yet. What the package owes the caller is a clear statement of which answers are extrapolations
and how far out they reach.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.model import PredictionError
from simple_scaling_laws.simulate import simulate_runs

TRUE_LOSS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}


@pytest.fixture(scope="module")
def model():
    """A model fit over model sizes 1e6-1e8 and dataset sizes 1e3-1e5."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    return fit(frame, n_draws=200, seed=0)


def _predict(model, points):
    """Predict without letting the extrapolation warning become a test failure."""
    with pytest.warns(UserWarning) if _has_outside(model, points) else _no_warning():
        return model.predict(points)


def _has_outside(model, points):
    """Whether any requested point lies outside the observed domain."""
    raw = pl.DataFrame(points).select(model.predictors).to_numpy().astype(float)
    return bool(model.domain_position(raw)[0].any())


class _no_warning:
    """A context manager that asserts no warning is raised."""

    def __enter__(self):
        import warnings

        self._catcher = warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        messages = [str(r.message) for r in self._records]
        self._catcher.__exit__(*exc)
        assert not messages, f"unexpected warning(s): {messages}"
        return False


def test_interior_points_are_labelled_interpolation(model):
    """A point inside the observed box on every predictor is interpolation, distance zero."""
    points = {"model_size__n_params": [1e7, 5e6], "dataset_size__n_subjects": [1e4, 2e4]}
    predictions = _predict(model, points)
    assert predictions["domain"].to_list() == ["interpolation", "interpolation"]
    assert predictions["extrapolation_distance"].to_list() == [0.0, 0.0]


def test_domain_corners_count_as_interpolation(model):
    """The observed extremes themselves are inside the domain, not outside it."""
    points = {"model_size__n_params": [1e6, 1e8], "dataset_size__n_subjects": [1e3, 1e5]}
    predictions = _predict(model, points)
    assert predictions["domain"].to_list() == ["interpolation", "interpolation"]


def test_points_beyond_the_domain_are_labelled_extrapolation(model):
    """Exceeding the observed range on any single predictor is enough to be extrapolation."""
    points = {
        "model_size__n_params": [1e9, 1e7, 1e5],
        "dataset_size__n_subjects": [1e4, 1e6, 1e4],
    }
    predictions = _predict(model, points)
    assert predictions["domain"].to_list() == ["extrapolation"] * 3


def test_extrapolation_distance_is_measured_in_observed_log_ranges(model):
    """One full observed range beyond the edge is a distance of exactly one."""
    # Model size spans 1e6 to 1e8, i.e. two decades; 1e10 is two decades beyond the top.
    points = {"model_size__n_params": [1e10, 1e9], "dataset_size__n_subjects": [1e4, 1e4]}
    predictions = _predict(model, points)
    assert predictions["extrapolation_distance"].to_list() == pytest.approx([1.0, 0.5])


def test_distance_takes_the_worst_predictor(model):
    """The reported distance is the largest excursion across predictors."""
    points = {"model_size__n_params": [1e9], "dataset_size__n_subjects": [1e7]}
    predictions = _predict(model, points)
    assert predictions["extrapolation_distance"][0] == pytest.approx(1.0)


def test_extrapolation_raises_a_warning(model):
    """Callers who are not reading the column still get told."""
    points = {"model_size__n_params": [1e12], "dataset_size__n_subjects": [1e4]}
    with pytest.warns(UserWarning, match="outside the observed scaling domain"):
        model.predict(points)


def test_interpolation_raises_no_warning(model):
    """Ordinary interior predictions are silent."""
    points = {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]}
    _predict(model, points)


def test_extrapolated_intervals_are_wider_than_interior_ones(model):
    """Uncertainty grows as the question moves away from the evidence."""
    points = {
        "model_size__n_params": [1e7, 1e9, 1e11],
        "dataset_size__n_subjects": [1e4, 1e4, 1e4],
    }
    predictions = _predict(model, points)
    widths = (
        predictions["test_loss__ce__q975"] - predictions["test_loss__ce__q025"]
    ).to_list()
    assert widths[0] < widths[1] < widths[2]


def test_a_predictor_held_fixed_makes_any_other_value_extrapolation():
    """If dataset size never varied, no value of it other than the observed one is supported."""
    frame = simulate_runs(
        {"test_loss__ce": TRUE_LOSS},
        model_sizes=(1e6, 1e7, 1e8, 1e9),
        dataset_sizes=(1e4,),
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )
    model = fit(frame, n_draws=100, seed=0)
    assert model.fitted_predictors == ("model_size__n_params",)
    assert model.predictors == ("model_size__n_params", "dataset_size__n_subjects")

    points = {"model_size__n_params": [1e7, 1e7], "dataset_size__n_subjects": [1e4, 1e5]}
    with pytest.warns(UserWarning):
        predictions = model.predict(points)
    assert predictions["domain"].to_list() == ["interpolation", "extrapolation"]
    assert np.isinf(predictions["extrapolation_distance"][1])
    codes = {note.code for note in model.warnings}
    assert "constant_predictor_dropped" in codes


def test_prediction_points_must_supply_every_predictor(model):
    """A missing predictor column is an error, not a silently substituted default."""
    with pytest.raises(PredictionError, match="missing predictor column"):
        model.predict({"model_size__n_params": [1e7]})


def test_prediction_points_must_be_positive(model):
    """Predictors are log-scaled, so zero and negative values are rejected."""
    with pytest.raises(PredictionError, match="strictly positive"):
        model.predict({"model_size__n_params": [0.0], "dataset_size__n_subjects": [1e4]})


def test_point_ids_default_to_row_positions(model):
    """Callers who do not supply identifiers still get stable ones."""
    points = {"model_size__n_params": [1e7, 2e7], "dataset_size__n_subjects": [1e4, 1e4]}
    assert _predict(model, points)["point_id"].to_list() == ["0", "1"]


def test_supplied_point_ids_are_carried_through(model):
    """Supplied identifiers survive so predictions can be joined back to their requests."""
    points = {
        "point_id": ["hypothesis-a", "hypothesis-b"],
        "model_size__n_params": [1e7, 2e7],
        "dataset_size__n_subjects": [1e4, 1e4],
    }
    assert _predict(model, points)["point_id"].to_list() == ["hypothesis-a", "hypothesis-b"]
