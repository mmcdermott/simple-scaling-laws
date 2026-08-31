"""Property-based tests of the numerical and structural invariants.

These check the things that must hold for *every* input rather than for a chosen example: the analytic
derivatives really are the derivatives, the vectorized and scalar evaluators agree, weights stay bounded,
serialization is total, and a loss law really is monotone in scale.
"""

import json
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from simple_scaling_laws.artifact import jsonable
from simple_scaling_laws.fitting import (
    MAX_WEIGHT_RATIO,
    Bounds,
    compute_weights,
    is_constant,
    parameter_bounds,
    replicate_run_variance,
)
from simple_scaling_laws.laws import MAX_EXPONENT, available_laws, build_law
from simple_scaling_laws.model import NAMED_QUANTILES, quantile_suffix
from simple_scaling_laws.schema import discover_schema

LAW_NAMES = sorted(available_laws())

finite_floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32)
positive_scales = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
exponents = st.floats(min_value=0.0, max_value=MAX_EXPONENT, allow_nan=False, allow_infinity=False)


def _law(name, n_model, n_dataset):
    """Build a law over synthetic predictor names."""
    return build_law(
        name,
        [f"model_size__m{i}" for i in range(n_model)],
        [f"dataset_size__d{i}" for i in range(n_dataset)],
    )


@given(
    name=st.sampled_from(LAW_NAMES),
    n_model=st.integers(1, 2),
    n_dataset=st.integers(0, 2),
    scales=st.lists(positive_scales, min_size=4, max_size=24),
    seed=st.integers(0, 2**16),
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_analytic_jacobian_matches_finite_differences(name, n_model, n_dataset, scales, seed):
    """The hand-written derivatives must agree with numerical ones."""
    law = _law(name, n_model, n_dataset)
    rng = np.random.default_rng(seed)
    n_points = max(3, len(scales) // max(1, law.n_exponents))
    log_x = np.log(rng.choice(np.asarray(scales), size=(n_points, len(law.predictors))))
    params = np.concatenate([rng.uniform(-2.0, 2.0, law.n_linear), rng.uniform(0.0, 1.5, law.n_exponents)])

    analytic = law.jacobian(params, log_x)
    numerical = np.zeros_like(analytic)
    for j in range(law.n_params):
        step = 1e-6 * max(1.0, abs(params[j]))
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (law.evaluate(up, log_x) - law.evaluate(down, log_x)) / (2 * step)
    scale = np.maximum(np.abs(numerical).max(), 1.0)
    assert np.allclose(analytic, numerical, atol=1e-4 * scale, rtol=1e-4)


@given(
    name=st.sampled_from(LAW_NAMES),
    n_model=st.integers(1, 2),
    n_dataset=st.integers(0, 2),
    n_draws=st.integers(1, 6),
    seed=st.integers(0, 2**16),
)
@settings(max_examples=60, deadline=None)
def test_vectorized_evaluation_matches_the_scalar_one(name, n_model, n_dataset, n_draws, seed):
    """``evaluate_many`` is what prediction uses; it must equal looping over ``evaluate``."""
    law = _law(name, n_model, n_dataset)
    rng = np.random.default_rng(seed)
    log_x = np.log(rng.uniform(0.05, 20.0, size=(5, len(law.predictors))))
    draws = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, size=(n_draws, law.n_linear)),
            rng.uniform(0.0, 1.5, size=(n_draws, law.n_exponents)),
        ]
    )
    expected = np.array([law.evaluate(p, log_x) for p in draws])
    assert np.allclose(law.evaluate_many(draws, log_x), expected, rtol=1e-12, atol=1e-12)


@given(
    offset=finite_floats,
    amplitudes=st.lists(st.floats(0.0, 50.0, allow_nan=False, width=32), min_size=1, max_size=2),
    exponent_values=st.lists(exponents, min_size=1, max_size=2),
    low=positive_scales,
    ratio=st.floats(1.01, 1e4, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_a_loss_law_never_increases_with_scale(offset, amplitudes, exponent_values, low, ratio):
    """Non-negative amplitudes and exponents are exactly what make a loss law monotone."""
    n = min(len(amplitudes), len(exponent_values))
    law = _law("separable-power", n, 0)
    params = np.array([offset, *amplitudes[:n], *exponent_values[:n]])
    small = np.log(np.full((1, n), low))
    large = np.log(np.full((1, n), low * ratio))
    assert law.evaluate(params, large)[0] <= law.evaluate(params, small)[0] + 1e-9


@given(
    values=st.lists(finite_floats, min_size=2, max_size=30),
    signed=st.booleans(),
    name=st.sampled_from(LAW_NAMES),
)
@settings(max_examples=100, deadline=None)
def test_parameter_bounds_are_ordered_and_contain_a_sensible_point(values, signed, name):
    """Bounds must be a real box, and must admit the constant fit at the data's own level."""
    law = _law(name, 1, 1)
    y = np.array(values)
    bounds = parameter_bounds(law, y, signed_amplitude=signed)
    assert np.all(bounds.lower <= bounds.upper)
    assert np.all(np.isfinite(bounds.lower)) and np.all(np.isfinite(bounds.upper))
    constant = np.zeros(law.n_params)
    constant[0] = float(np.mean(y))
    assert np.allclose(bounds.clip(constant), constant)


@given(
    values=st.lists(finite_floats, min_size=2, max_size=20),
    params=st.lists(finite_floats, min_size=5, max_size=5),
)
@settings(max_examples=100, deadline=None)
def test_clipping_is_idempotent_and_lands_inside_the_box(values, params):
    """Clipping twice is the same as clipping once, and the result is always feasible."""
    law = _law("separable-power", 1, 1)
    bounds = parameter_bounds(law, np.array(values), signed_amplitude=True)
    once = bounds.clip(np.array(params))
    assert np.all(once >= bounds.lower) and np.all(once <= bounds.upper)
    assert np.array_equal(bounds.clip(once), once)


@given(
    run_var=st.floats(0.0, 10.0, allow_nan=False, width=32),
    per_run_var=st.lists(st.floats(0.0, 10.0, allow_nan=False, width=32), min_size=1, max_size=20),
)
@settings(max_examples=200, deadline=None)
def test_weights_are_positive_normalized_and_bounded(run_var, per_run_var):
    """Weights average to one and cannot span more than the documented ratio."""
    weights = compute_weights(run_var, np.array(per_run_var))
    assert np.all(weights > 0)
    assert np.all(np.isfinite(weights))
    assert weights.mean() == pytest.approx(1.0)
    assert weights.max() / weights.min() <= MAX_WEIGHT_RATIO * (1 + 1e-9)


@given(
    run_var=st.floats(0.0, 10.0, allow_nan=False, width=32),
    per_run_var=st.lists(st.floats(0.0, 10.0, allow_nan=False, width=32), min_size=2, max_size=20),
)
@settings(max_examples=200, deadline=None)
def test_a_more_precisely_measured_run_is_never_down_weighted(run_var, per_run_var):
    """Weights must be non-increasing in a run's own noise."""
    values = np.array(per_run_var)
    weights = compute_weights(run_var, values)
    order = np.argsort(values)
    ranked = weights[order]
    assert np.all(np.diff(ranked) <= 1e-9)


@given(
    means=st.lists(finite_floats, min_size=2, max_size=20),
    n_eval=st.integers(1, 50),
    eval_var=st.floats(0.0, 5.0, allow_nan=False, width=32),
)
@settings(max_examples=150, deadline=None)
def test_replicate_run_variance_is_never_negative(means, n_eval, eval_var):
    """A variance estimate is floored at zero however the noise correction lands."""
    from simple_scaling_laws.data import TargetObservations

    values = np.array(means)
    observations = TargetObservations(
        target="t",
        run_ids=tuple(f"r{i}" for i in range(values.size)),
        config_index=np.zeros(values.size, dtype=int),
        mean=values,
        n_eval=np.full(values.size, n_eval),
        within_ss=0.0,
        within_dof=values.size * (n_eval - 1),
        eval_pair_correlation=None,
        n_shared_pairs=0,
    )
    estimate, dof = replicate_run_variance(observations, eval_var)
    assert estimate is not None and estimate >= 0.0
    assert dof == values.size - 1


@given(q=st.floats(0.0, 1.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_quantile_suffixes_are_well_formed(q):
    """Every quantile gets a column-name-safe suffix, and only the endpoints get words."""
    suffix = quantile_suffix(q)
    assert suffix.isidentifier()
    assert (suffix in NAMED_QUANTILES.values()) == any(abs(q - value) < 5e-4 for value in NAMED_QUANTILES)


@given(quantiles=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=8, unique=True))
@settings(max_examples=100, deadline=None)
def test_distinct_quantiles_get_distinct_suffixes(quantiles):
    """Two quantiles that differ at three decimals must not collide into one column."""
    rounded = {round(q, 3) for q in quantiles}
    assume(len(rounded) == len(quantiles))
    suffixes = {quantile_suffix(q) for q in quantiles}
    assert len(suffixes) == len(quantiles)


json_values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-(2**40), 2**40),
        st.floats(allow_nan=True, allow_infinity=True, width=32),
        st.text(max_size=10),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4), st.dictionaries(st.text(max_size=5), children, max_size=4)
    ),
    max_leaves=12,
)


@given(value=json_values)
@settings(max_examples=200, deadline=None)
def test_jsonable_output_is_always_strictly_serializable(value):
    """Artifacts must never contain bare NaN or Infinity, which are not valid JSON."""
    text = json.dumps(jsonable(value), allow_nan=False)
    json.loads(text, parse_constant=lambda token: pytest.fail(f"invalid token {token}"))


@given(value=st.floats(allow_nan=True, allow_infinity=True, width=32))
@settings(max_examples=100, deadline=None)
def test_non_finite_floats_become_null(value):
    """Non-finite values are represented as null rather than dropped or stringified."""
    converted = jsonable(value)
    assert converted is None if not math.isfinite(value) else converted == pytest.approx(value)


@given(
    suffixes=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=6),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    extra=st.lists(st.text(alphabet="XYZ", min_size=1, max_size=4), max_size=3, unique=True),
)
@settings(max_examples=100, deadline=None)
def test_discovery_finds_every_prefixed_column_and_ignores_the_rest(suffixes, extra):
    """Role assignment depends only on the prefix, whatever else the table contains."""
    assume(not any(e.startswith(("model_size__", "test_loss__")) for e in extra))
    columns = ["training_run_id"]
    columns += [f"model_size__{s}" for s in suffixes]
    columns += [f"dataset_size__{s}" for s in suffixes]
    columns += [f"test_loss__{s}" for s in suffixes]
    columns += extra
    schema = discover_schema(columns)
    assert len(schema.model_size) == len(suffixes)
    assert len(schema.dataset_size) == len(suffixes)
    assert len(schema.targets) == len(suffixes)
    assert schema.primary_target.startswith("test_loss__")
    assert all(e not in schema.target_names and e not in schema.predictors for e in extra)


@given(values=st.lists(finite_floats, min_size=1, max_size=20))
@settings(max_examples=200, deadline=None)
def test_constant_detection_agrees_with_the_observed_spread(values):
    """A target is constant exactly when its values do not move."""
    array = np.array(values)
    if is_constant(array):
        assert np.ptp(array) <= 1e-10 * max(1.0, float(np.abs(array).max()))
    else:
        assert np.ptp(array) > 0


@given(name=st.sampled_from(LAW_NAMES), n_model=st.integers(1, 3), n_dataset=st.integers(0, 3))
@settings(max_examples=100, deadline=None)
def test_law_structure_is_self_consistent_and_round_trips(name, n_model, n_dataset):
    """Names, kinds and the linear/nonlinear split must stay in agreement, and survive saving."""
    from simple_scaling_laws.laws import LawInstance

    law = _law(name, n_model, n_dataset)
    assert len(law.param_names) == len(law.param_kinds) == law.n_params
    assert len(set(law.param_names)) == law.n_params
    assert law.n_linear + law.n_exponents == law.n_params
    assert law.n_exponents == n_model + n_dataset
    assert sorted(law.display_names) == sorted(law.param_names)
    assert LawInstance.from_dict(law.to_dict()) == law


@given(
    lower=st.lists(finite_floats, min_size=3, max_size=3),
    widths=st.lists(st.floats(0.0, 100.0, allow_nan=False, width=32), min_size=3, max_size=3),
)
@settings(max_examples=200, deadline=None)
def test_at_bound_only_reports_parameters_on_a_boundary(lower, widths):
    """A parameter strictly inside the box is never called pinned."""
    low = np.array(lower)
    high = low + np.array(widths)
    bounds = Bounds(low, high)
    names = ("a", "b", "c")
    midpoints = (low + high) / 2
    interior = [n for n, w in zip(names, widths, strict=True) if w > 1e-3]
    assert set(bounds.at_bound(midpoints, names)).isdisjoint(interior)
    assert set(bounds.at_bound(low, names)) == set(names)
