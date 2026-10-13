"""Focused tests for the separable interaction power law."""

import numpy as np

from simple_scaling_laws.laws import build_law
from simple_scaling_laws.model import raw_scale_params


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
