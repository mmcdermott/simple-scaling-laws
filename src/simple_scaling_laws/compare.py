"""Comparing two or more fitted systems at scales none of them has been run at.

Fitting a scaling law per system answers "how does each one scale". The question a triage platform
actually asks is comparative: *of these arms, which is worth scaling up, and how sure are we?* That
needs the distribution of the **difference**, not two separate intervals -- two intervals that
overlap are perfectly compatible with one system being reliably better.

The difference has to be computed draw by draw, and the draws have to be *paired*. Arms run at the
same configurations share a configuration-level deviation: a split that is hard for one is hard for
the other, and a place where the law misfits bends both curves the same way. That shared part
cancels in the difference, so the difference is pinned down far more sharply than either level is.

The wild bootstrap makes pairing almost free. It perturbs residuals with a multiplier drawn once per
configuration; if every arm gets the **same** multiplier sequence, the perturbation of the difference
is ``w_c * (e_c^A - e_c^B)`` -- driven by the difference's own residual. That is right whether or not
the arms actually share splits: correlated residuals give a small difference and a sharp interval,
independent ones give ``E[(e^A - e^B)^2] = Var(e^A) + Var(e^B)`` and recover the independent answer.
The pairing lives in the observed residuals; the multipliers only have to be common.

Measured on synthetic arms differing only in their exponent, against the true sampling spread of the
difference at ten times the observed scale:

===========================  ==============  ==============  =================
shared configuration sd      true sd(A-B)    paired draws    independent draws
===========================  ==============  ==============  =================
0.00                         0.0329          0.0413 (1.26x)  0.0430 (1.31x)
0.05                         0.0330          0.0416 (1.26x)  0.1330 (4.03x)
0.15                         0.0338          0.0442 (1.31x)  0.3715 (11.00x)
===========================  ==============  ==============  =================

Unpaired differencing degrades to four and eleven times too wide -- wide enough that no real
difference is ever detectable. Pairing holds at the package's ordinary small-sample factor.

Fits pair automatically when they were run with the same ``seed`` on the same configuration grid,
which is the normal case for arms of one experiment. When they cannot be verified as comparable the
comparison still runs, marked ``paired=False``, since an unpaired interval is conservative rather
than wrong.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .metrics import HIGHER
from .model import DEFAULT_QUANTILES, POINT_ID, PREDICTION_KINDS, PredictionError, quantile_suffix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from .model import ScalingLawModel


class ComparisonError(ValueError):
    """Raised when a set of fitted models cannot be compared."""


def _as_mapping(models: Mapping[str, ScalingLawModel] | Sequence[ScalingLawModel]) -> dict:
    """Accept either a name-to-model mapping or a bare sequence of models."""
    if isinstance(models, dict):
        named = dict(models)
    else:
        named = {f"system_{i}": model for i, model in enumerate(models)}
    if len(named) < 2:
        raise ComparisonError(f"Comparing needs at least two systems, got {len(named)}")
    return named


def _resolve_target(named: dict, target: str | None) -> str:
    """Pick the target to compare on, and check every system actually fit it."""
    if target is None:
        primaries = {model.primary_target for model in named.values()}
        if len(primaries) != 1:
            raise ComparisonError(
                f"The systems have different primary targets {sorted(primaries)}; name the one to "
                "compare on with target=..."
            )
        target = primaries.pop()
    missing = [name for name, model in named.items() if target not in model.targets]
    if missing:
        raise ComparisonError(f"System(s) {missing} have no fit for target {target!r}")
    return target


def _check_comparable(named: dict, target: str) -> None:
    """Refuse comparisons that are not meaningful, whatever the pairing status."""
    laws = {model.law.law for model in named.values()}
    if len(laws) != 1:
        raise ComparisonError(f"The systems were fit with different laws {sorted(laws)}")
    predictors = {model.fitted_predictors for model in named.values()}
    if len(predictors) != 1:
        raise ComparisonError(
            f"The systems were fit over different predictors {sorted(map(list, predictors))}"
        )
    kinds = {model.target_kind(target).support for model in named.values()}
    if len(kinds) != 1:
        raise ComparisonError(
            f"Target {target!r} was fit on different scales across systems ({sorted(kinds)}); the "
            "values are not on a common footing and differencing them would be meaningless."
        )


def _pairing_status(named: dict, target: str) -> tuple[bool, str | None]:
    """Whether every system's draws for this target used the same cluster multipliers."""
    keys = {name: model.fits[target].pairing for name, model in named.items()}
    if any(not key for key in keys.values()):
        return False, "one or more fits predate the recording of pairing information"
    distinct = {tuple(sorted(key.items())) for key in keys.values()}
    if len(distinct) != 1:
        return False, "the fits used different bootstrap seeds, draw counts or configuration counts"
    grids = {
        tuple(sorted(tuple(sorted(c.items())) for c in model.manifest.get("configurations", [])))
        for model in named.values()
    }
    if len(grids) != 1:
        return False, "the systems were run at different scaling configurations"
    if next(iter(keys.values()))["method"] != "wild-cluster":
        return False, "the draws did not come from the wild cluster bootstrap"
    return True, None


def _draw_index(n_draws: int, n_systems: int, paired: bool, rng: np.random.Generator) -> list[np.ndarray]:
    """Which draw each system contributes to each comparison.

    Paired comparisons line draw ``b`` of one system up with draw ``b`` of every other, because those draws
    share a cluster multiplier sequence. Unpaired ones resample independently, which is the honest thing to do
    when nothing ties the two bootstraps together.
    """
    if paired:
        return [np.arange(n_draws)] * n_systems
    return [rng.permutation(n_draws) for _ in range(n_systems)]


def compare(
    models: Mapping[str, ScalingLawModel] | Sequence[ScalingLawModel],
    points: Any,
    target: str | None = None,
    reference: str | None = None,
    kind: str = "mean",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    seed: int = 0,
) -> pl.DataFrame:
    """Compare fitted systems at new scales, with a distribution over their differences.

    Args:
        models: Two or more fitted models, as ``{name: model}`` or a bare sequence.
        points: The scales to compare at, as for
            :meth:`~simple_scaling_laws.model.ScalingLawModel.predict`.
        target: Which target to compare on. Defaults to the systems' shared primary target.
        reference: The system to measure differences against. Defaults to no reference, in which
            case only each system's own prediction and its probability of being best are reported.
        kind: ``"mean"`` to compare the expected scaling curves, or ``"new-run"`` to compare single
            newly trained models, which additionally carries each arm's training stochasticity.
        quantiles: Quantiles to report, for both the values and the differences.
        seed: Seed for the ``new-run`` noise and for unpaired draw resampling.

    Returns:
        One row per requested point and system, with that system's predicted value, its probability
        of being the best system at that point, and -- when ``reference`` is given -- the
        distribution of its difference from the reference and the probability it is better.

    Raises:
        ComparisonError: If fewer than two systems are given, or they were fit in ways that cannot
            be compared.
        PredictionError: If ``kind`` is unknown or the points are malformed.

    Examples:
        >>> from simple_scaling_laws import compare, fit
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> def arm(alpha, seed):
        ...     return simulate_runs(
        ...         {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": alpha, "B": 1.5, "beta": 0.25}},
        ...         runs_per_config=2,
        ...         evaluations_per_run=4,
        ...         run_sd=0.01,
        ...         eval_sd=0.02,
        ...         seed=seed,
        ...     )
        >>> baseline = fit(arm(0.30, 0), n_draws=200, seed=0)
        >>> improved = fit(arm(0.34, 1), n_draws=200, seed=0)
        >>> points = {"model_size__n_params": [1e9], "dataset_size__n_subjects": [1e4]}
        >>> table = compare(
        ...     {"baseline": baseline, "improved": improved},
        ...     points,
        ...     reference="baseline",
        ... )
        >>> table.select("system", "paired", "p_best", "p_better_than_reference")
        shape: (2, 4)
        ┌──────────┬────────┬────────┬─────────────────────────┐
        │ system   ┆ paired ┆ p_best ┆ p_better_than_reference │
        │ ---      ┆ ---    ┆ ---    ┆ ---                     │
        │ str      ┆ bool   ┆ f64    ┆ f64                     │
        ╞══════════╪════════╪════════╪═════════════════════════╡
        │ baseline ┆ true   ┆ 0.0    ┆ 0.0                     │
        │ improved ┆ true   ┆ 1.0    ┆ 1.0                     │
        └──────────┴────────┴────────┴─────────────────────────┘

        The arm with the steeper exponent wins at ten times the observed model size with probability
        one. The size of the win, and its interval, come back alongside:

        >>> row = table.filter(pl.col("system") == "improved").row(0, named=True)
        >>> round(row["difference__median"], 3) < 0  # a loss, so a negative difference is better
        True
    """
    if kind not in PREDICTION_KINDS:
        raise PredictionError(f"kind must be one of {list(PREDICTION_KINDS)}, got {kind!r}")
    named = _as_mapping(models)
    target = _resolve_target(named, target)
    _check_comparable(named, target)
    if reference is not None and reference not in named:
        raise ComparisonError(f"Reference {reference!r} is not one of the systems {list(named)}")

    quantiles = [float(q) for q in quantiles]
    if any(not 0 <= q <= 1 for q in quantiles):
        raise ComparisonError(f"quantiles must lie in [0, 1], got {quantiles}")

    paired, reason = _pairing_status(named, target)
    if not paired:
        warnings.warn(
            f"The systems' draws for {target!r} could not be paired because {reason}. The "
            "comparison is still valid but its intervals are conservative -- often several times "
            "too wide, since the shared uncertainty does not cancel. Refit every system with the "
            "same seed on the same scaling configurations to pair them.",
            UserWarning,
            stacklevel=2,
        )

    kinds = {name: model.target_kind(target) for name, model in named.items()}
    direction = next(iter(kinds.values())).direction
    known_direction = all(k.known_direction for k in kinds.values())
    if not known_direction:
        warnings.warn(
            f"Which direction is better for {target!r} is not known, so it is assumed that "
            f"{direction} is better. Say so explicitly with "
            f'targets={{{target!r}: {{"direction": "higher"}}}} at fit time if that is wrong.',
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    first = next(iter(named.values()))
    frame = first.prepare_points(points)
    n_points = frame.height
    raw = frame.select(first.predictors).to_numpy().astype(float)

    n_draws = min(model.draws[target].n_draws for model in named.values())
    index = _draw_index(n_draws, len(named), paired, rng)

    values: dict[str, np.ndarray] = {}
    extrapolating = np.zeros(n_points, dtype=bool)
    distances = np.zeros(n_points, dtype=float)
    for position, (name, model) in enumerate(named.items()):
        drawn = model.target_draws(target, model.fitting_predictors(frame), kind=kind, rng=rng)
        values[name] = drawn[index[position][:n_draws]]
        outside, distance = model.domain_position(raw)
        extrapolating |= outside
        distances = np.maximum(distances, distance)
    if bool(extrapolating.any()):
        warnings.warn(
            f"{int(extrapolating.sum())} of {n_points} comparison point(s) lie outside at least one "
            "system's observed scaling domain. Which system wins out there rests on the fitted "
            "curves continuing to hold, which the data cannot speak to.",
            UserWarning,
            stacklevel=2,
        )

    stacked = np.stack([values[name] for name in named])  # (n_systems, n_draws, n_points)
    best = stacked.min(axis=0) if direction != HIGHER else stacked.max(axis=0)
    is_best = stacked == best[None, :, :]
    # Split the credit on an exact tie rather than handing it to whichever system is listed first,
    # which matters when two arms are genuinely indistinguishable.
    p_best = (is_best / is_best.sum(axis=0, keepdims=True)).mean(axis=1)

    rows: list[pl.DataFrame] = []
    for position, name in enumerate(named):
        columns: dict[str, Any] = {
            POINT_ID: frame[POINT_ID],
            **{predictor: frame[predictor] for predictor in first.predictors},
            "target": np.full(n_points, target),
            "system": np.full(n_points, name),
            "domain": np.where(extrapolating, "extrapolation", "interpolation"),
            "extrapolation_distance": distances,
            "paired": np.full(n_points, paired),
            "direction": np.full(n_points, direction),
            "p_best": p_best[position],
        }
        for q in quantiles:
            columns[f"value__{quantile_suffix(q)}"] = np.quantile(values[name], q, axis=0)
        if reference is not None:
            difference = values[name] - values[reference]
            for q in quantiles:
                columns[f"difference__{quantile_suffix(q)}"] = np.quantile(difference, q, axis=0)
            better = difference < 0 if direction != HIGHER else difference > 0
            columns["p_better_than_reference"] = better.mean(axis=0)
        rows.append(pl.DataFrame(columns))
    return pl.concat(rows).sort(POINT_ID, "system")
