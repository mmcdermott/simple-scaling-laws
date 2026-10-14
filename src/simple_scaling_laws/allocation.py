"""Compute-optimal allocation: the best model size for a budget, read off a fitted law.

Given a fitted scaling law ``L(N, D)`` and a cost law ``C(N, D)`` (:mod:`simple_scaling_laws.cost`), the best
use of a budget ``C`` is the point on the curve of equal cost where the law is lowest -- Chinchilla's
"Approach 3". This module finds that point for **every bootstrap draw** of the law, so the optimal model size,
dataset size and loss come with the same intervals as any other prediction, and with no new fitting: it is a
summary of draws the fit already stored.

The search runs along the iso-cost curve over a window that extends a few decades beyond the observed range in
both ``N`` and ``D``. A draw whose minimum lies at an end of that window has no interior optimum: for that
draw, putting everything into one resource keeps paying. ``multiplicative-power``, which is monotone along any
iso-cost curve, always ends up there. The fraction of such draws is reported, and the optimum is *not*
trustworthy when it is large.

What this cannot tell you is whether the law holds out at the optimum. An optimum outside the observed range
is an extrapolation of the law's shape, and is flagged as such; the law-free check is the isoFLOP profile fit
in :mod:`simple_scaling_laws.isoflop`.

**Sources.** Minimizing a fitted law along the iso-cost curve is Chinchilla's Approach 3 (Hoffmann et al.,
2022). Besiroglu et al. (2024), *Chinchilla Scaling: A replication attempt*, and Czech et al. (2026),
*Problems with Chinchilla Approach 2*, argue for it over Approach 2 when the law is right; Czech et al.
recommend fitting it by variable projection with grid initialization, which is how this package fits every
law. Taking intervals by minimizing each bootstrap draw is package-specific.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .model import DEFAULT_QUANTILES, PredictionError, quantile_suffixes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .cost import CostModel
    from .model import ScalingLawModel

#: How far beyond the observed range of each predictor, in decades, the optimum is searched for.
SEARCH_DECADES = 3.0

#: Points on the iso-cost curve at which every draw is evaluated before refining its minimum.
SEARCH_POINTS = 801

#: Above this fraction of draws without an interior optimum, a warning is raised.
BOUNDARY_WARNING_FRACTION = 0.05


def _predictor_pair(model: ScalingLawModel) -> tuple[str, str]:
    law = model.law
    if len(law.model_size) != 1 or len(law.dataset_size) != 1:
        raise PredictionError(
            "Allocating a budget trades one model-size predictor against one dataset-size predictor, but "
            f"this law was fit over {list(law.model_size)} and {list(law.dataset_size)}"
        )
    return law.model_size[0], law.dataset_size[0]


def _search_curve(model: ScalingLawModel, cost: CostModel, budget: float) -> tuple[np.ndarray, np.ndarray]:
    """Model and dataset sizes along the iso-cost curve, inside the search window of both."""
    n_col, d_col = _predictor_pair(model)
    domain = model.observed_domain
    widen = SEARCH_DECADES * np.log(10.0)
    n_low, n_high = np.log(domain[n_col]["min"]) - widen, np.log(domain[n_col]["max"]) + widen
    d_low, d_high = np.log(domain[d_col]["min"]) - widen, np.log(domain[d_col]["max"]) + widen
    n = np.exp(np.linspace(n_low, n_high, SEARCH_POINTS))
    d = cost.dataset_size_for(budget, n)
    inside = (np.log(d) >= d_low) & (np.log(d) <= d_high)
    return n[inside], d[inside]


def _refine(log_n: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each draw's minimum along the curve, refined between grid points by a parabola.

    Returns the log model size and value at each draw's minimum, and whether it sits at an end.
    """
    n_draws, n_points = values.shape
    best = np.argmin(values, axis=1)
    boundary = (best == 0) | (best == n_points - 1)
    inner = np.clip(best, 1, n_points - 2)
    rows = np.arange(n_draws)
    left, centre, right = values[rows, inner - 1], values[rows, inner], values[rows, inner + 1]
    curvature = left - 2.0 * centre + right
    safe = np.where(curvature > 0, curvature, 1.0)
    offset = np.where(curvature > 0, 0.5 * (left - right) / safe, 0.0)
    offset = np.where(boundary, 0.0, np.clip(offset, -0.5, 0.5))
    step = log_n[1] - log_n[0]
    position = log_n[best] + np.where(boundary, 0.0, offset * step)
    value = np.where(boundary, values[rows, best], centre - 0.25 * (left - right) * offset)
    return position, value, boundary


def _optimum_draws(
    model: ScalingLawModel, cost: CostModel, budget: float, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per draw: optimal model size, dataset size, target value, and whether the optimum is interior."""
    n, d = _search_curve(model, cost, budget)
    n_draws = model.draws[target].n_draws
    if n.size < 3:
        nan = np.full(n_draws, np.nan)
        return nan, nan, nan, np.ones(n_draws, dtype=bool)
    frame = pl.DataFrame(dict(zip(_predictor_pair(model), (n, d), strict=True)))
    values = model.target_draws(target, model.fitting_predictors(frame))
    sign = -1.0 if model.target_kind(target).direction == "higher" else 1.0
    log_n, value, boundary = _refine(np.log(n), sign * values)
    n_star = np.exp(log_n)
    return n_star, cost.dataset_size_for(budget, n_star), sign * value, boundary


def optimal_allocation(
    model: ScalingLawModel,
    budgets: Sequence[float],
    cost: CostModel,
    targets: Sequence[str] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pl.DataFrame:
    """The best model size, dataset size and target value at each budget, with intervals.

    Args:
        model: A fitted model over one model-size and one dataset-size predictor.
        budgets: Budgets, in the units of ``cost``.
        cost: The cost law, see :func:`simple_scaling_laws.cost.cost_model` and
            :func:`simple_scaling_laws.cost.fit_cost_model`.
        targets: Targets to optimize, one at a time; by default the primary target. A target that is
            better when higher is maximized.
        quantiles: Quantiles of each quantity to report.

    Returns:
        One row per target and budget, with quantiles of the optimal model size and dataset size
        (named after their columns) and of the target at the optimum; ``p_no_interior_optimum``, the
        fraction of draws whose optimum is at an end of the search window; and ``domain`` and
        ``extrapolation_distance`` of the median optimum, as in
        :meth:`~simple_scaling_laws.model.ScalingLawModel.predict`.

    Raises:
        PredictionError: If the law is not over exactly one predictor of each role, or a budget or
            quantile is invalid.

    Examples:
        >>> from simple_scaling_laws import fit
        >>> from simple_scaling_laws.cost import cost_model
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> truth = {"E": 1.0, "A": 2.0, "alpha": 0.35, "B": 2.0, "beta": 0.35}
        >>> runs = simulate_runs(
        ...     {"test_loss__ce": truth},
        ...     model_sizes=(1e6, 1e7, 1e8, 1e9),
        ...     dataset_sizes=(1e7, 1e8, 1e9, 1e10),
        ...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
        ...     run_sd=0.01,
        ...     eval_sd=0.01,
        ...     seed=0,
        ... )
        >>> model = fit(runs, n_draws=200, seed=0)
        >>> frontier = optimal_allocation(model, [1e16, 1e18], cost_model("power", k=6.0, a=1.0, b=1.0))
        >>> [f"{v:.2g}" for v in frontier["model_size__n_params__median"]]
        ['1.3e+07', '1.3e+08']

        With equal exponents in model and data, a hundred times the budget buys ten times the model and
        ten times the data:

        >>> [f"{v:.2g}" for v in frontier["dataset_size__n_tokens__median"]]
        ['1.3e+08', '1.3e+09']
    """
    n_col, d_col = _predictor_pair(model)
    budgets = [float(b) for b in budgets]
    if not budgets or not all(np.isfinite(b) and b > 0 for b in budgets):
        raise PredictionError(f"Budgets must be positive and finite, got {budgets}")
    quantiles = [float(q) for q in quantiles]
    if any(not 0 <= q <= 1 for q in quantiles):
        raise PredictionError(f"quantiles must lie in [0, 1], got {quantiles}")
    try:
        suffixes = quantile_suffixes(quantiles)
    except ValueError as error:
        raise PredictionError(str(error)) from None
    requested = list(targets) if targets is not None else [model.primary_target]
    unknown = [t for t in requested if t not in model.fits]
    if unknown:
        raise PredictionError(f"No fit for target(s) {unknown}; available: {list(model.targets)}")

    rows: list[dict[str, Any]] = []
    for target in requested:
        for budget in budgets:
            n_star, d_star, value, boundary = _optimum_draws(model, cost, budget, target)
            row: dict[str, Any] = {"target": target, "budget": budget}
            for column, draws in ((n_col, n_star), (d_col, d_star), (target, value)):
                for q, suffix in zip(quantiles, suffixes, strict=True):
                    row[f"{column}__{suffix}"] = (
                        float(np.nanquantile(draws, q)) if np.isfinite(draws).any() else None
                    )
            row["p_no_interior_optimum"] = float(boundary.mean())
            median = np.array([[float(np.nanmedian(n_star)), float(np.nanmedian(d_star))]])
            if np.all(np.isfinite(median)):
                is_out, distance = _domain_position(model, median)
                row["domain"] = "extrapolation" if bool(is_out[0]) else "interpolation"
                row["extrapolation_distance"] = float(distance[0])
            else:
                row["domain"], row["extrapolation_distance"] = "undefined", None
            rows.append(row)

    _warn(rows)
    return pl.DataFrame(rows)


def _domain_position(model: ScalingLawModel, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Domain position of (N, D) points, with any predictor the law dropped held at its one value."""
    pair = _predictor_pair(model)
    domain = model.observed_domain
    columns = []
    for name in domain:
        columns.append(
            points[:, pair.index(name)] if name in pair else np.full(points.shape[0], domain[name]["min"])
        )
    return model.domain_position(np.column_stack(columns))


def _warn(rows: list[dict[str, Any]]) -> None:
    flat = [r for r in rows if r["p_no_interior_optimum"] > BOUNDARY_WARNING_FRACTION]
    if flat:
        worst = max(r["p_no_interior_optimum"] for r in flat)
        warnings.warn(
            f"[no_interior_optimum] For {len(flat)} target/budget pair(s), up to {worst:.0%} of draws "
            f"keep improving all the way to the edge of the search ({SEARCH_DECADES:g} decades beyond the "
            "observed range): the law puts no limit on how much to spend on one resource, so the optimum "
            "there is not determined.",
            UserWarning,
            stacklevel=3,
        )
    outside = [r for r in rows if r["domain"] == "extrapolation"]
    if outside:
        warnings.warn(
            f"[optimum_outside_observed_domain] The optimum at {len(outside)} target/budget pair(s) lies "
            "outside the observed range of model or dataset size, so it extrapolates the law's shape, "
            "which the data cannot check.",
            UserWarning,
            stacklevel=3,
        )


def allocation_exponents(
    model: ScalingLawModel,
    budgets: Sequence[float],
    cost: CostModel,
    target: str | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, dict[str, float]]:
    """How fast the optimal model and dataset sizes grow with the budget, with intervals.

    For each draw, the slope of ``log N*`` and ``log D*`` against ``log C`` across ``budgets`` -- the
    ``a`` and ``b`` of ``N* ∝ C**a``, ``D* ∝ C**b``. Draws without an interior optimum at every budget
    are left out.

    Args:
        model: A fitted model over one model-size and one dataset-size predictor.
        budgets: At least two budgets spanning the range of interest.
        cost: The cost law.
        target: The target to optimize; by default the primary target.
        quantiles: Quantiles to report.

    Returns:
        ``{"model_size": {suffix: value}, "dataset_size": {suffix: value}}``.

    Examples:
        For the separable law with power cost ``C = k N D``, the optimal model size grows as
        ``C ** (beta / (alpha + beta))`` -- one half here, where ``alpha = beta``:

        >>> from simple_scaling_laws import fit
        >>> from simple_scaling_laws.cost import cost_model
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> truth = {"E": 1.0, "A": 2.0, "alpha": 0.35, "B": 2.0, "beta": 0.35}
        >>> runs = simulate_runs(
        ...     {"test_loss__ce": truth},
        ...     model_sizes=(1e6, 1e7, 1e8, 1e9),
        ...     dataset_sizes=(1e7, 1e8, 1e9, 1e10),
        ...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
        ...     run_sd=0.01,
        ...     eval_sd=0.01,
        ...     seed=0,
        ... )
        >>> model = fit(runs, n_draws=200, seed=0)
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> exponents = allocation_exponents(model, [1e15, 1e17, 1e19], chinchilla)
        >>> round(exponents["model_size"]["median"], 2)
        0.5
    """
    budgets = sorted(float(b) for b in budgets)
    if len(budgets) < 2:
        raise PredictionError("An allocation exponent needs at least two budgets")
    target = target or model.primary_target
    log_c = np.log(budgets)
    stars = [_optimum_draws(model, cost, budget, target) for budget in budgets]
    interior = ~np.any([boundary for *_, boundary in stars], axis=0)
    centred = log_c - log_c.mean()
    out: dict[str, dict[str, float]] = {}
    suffixes = quantile_suffixes([float(q) for q in quantiles])
    for role, index in (("model_size", 0), ("dataset_size", 1)):
        logs = np.log(np.column_stack([star[index] for star in stars]))[interior]
        slopes = (logs - logs.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)
        out[role] = {
            suffix: float(np.quantile(slopes, q)) if slopes.size else float("nan")
            for q, suffix in zip(quantiles, suffixes, strict=True)
        }
    return out
