"""IsoFLOP profiles: the compute-optimal model size, found without assuming a scaling law.

Train several model sizes at each of a few fixed budgets, and the loss at each budget traces a valley in
``log N``: too small a model underfits, too large a one sees too little data. Chinchilla's "Approach 2"
fits a parabola to each valley, takes its lowest point ``N*``, and fits a straight line of ``log N*``
against ``log C`` to extrapolate to budgets not yet run. Nothing in it depends on the functional form of a
scaling law -- only on each valley being smooth near its bottom -- which is what makes it the check on
:func:`~simple_scaling_laws.allocation.optimal_allocation`, which depends on the law entirely.

The two can disagree, and the disagreement is informative. Simulated with a law that was only slightly
wrong, the law's frontier was biased by a sixth of a decade in ``N*`` with an interval that never covered
the truth, while the parabolas stayed unbiased; with the law right, the law's frontier was the more
precise of the two. So report both (:meth:`IsoflopFit.compare`) rather than choosing.

**Uncertainty** comes from the same wild cluster bootstrap over configurations as every other fit here,
with the same residual corrections (:func:`~simple_scaling_laws.uncertainty.pseudo_residuals`), run through
*both* stages so that the noise in each valley reaches the extrapolated line. Each stage is linear in its
coefficients, so no optimizer is involved and the draws are exact refits. The minimum of a parabola is a
ratio of its coefficients, heavy-tailed when the valley is shallow, and a draw whose valley is not convex
has no minimum at all; both are reported rather than hidden.

**Why a cubic, not Chinchilla's parabola, by default.** Along a curve of equal cost the separable law is
``E + a * exp(-alpha * x) + b * exp(beta * x)`` in ``x = log N``: a valley that is lopsided whenever
``alpha != beta``. A parabola's minimum is then biased toward the shallow side, and resampling residuals
cannot see bias, so its intervals are too narrow exactly when the noise is small. Simulated over 200
experiments of five budgets with five to seven sizes each, the parabola's 95% intervals for ``N*`` at the
largest budget covered the truth only 48-72% of the time with low noise or wide sweeps (89% in the
easiest case); the cubic's covered 92-100% in every design. The cubic costs a parameter per profile and
gave intervals about twice as wide at moderate noise, and its minimum is less stable when the sweep barely
brackets it (reported as ``isoflop_unstable_minimum``). ``degree=2`` is there for a valley known to be
symmetric, or for comparison with published Approach-2 numbers.

**Design requirements.** Each point must be a run trained to its own budget, from scratch -- not an
intermediate checkpoint of a longer run -- which is already the package's one-observation-per-run rule.
Within one budget, model size and data move together exactly, by design. Each profile needs at least
``degree + 2`` model sizes -- five for the cubic -- since ``degree + 1`` determine it exactly and leave
nothing to judge it by, and its minimum should lie inside the sizes swept. The line needs at least three
budgets.

Budgets come from each run's recorded cost (a ``compute__*`` column) when there is one, and otherwise from
a :class:`~simple_scaling_laws.cost.CostModel` applied to its model and dataset size; runs whose budgets
agree to within :data:`BUDGET_TOLERANCE` form one profile.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .cost import COMPUTE_ROLE, CostModel
from .data import DataError, build_dataset
from .fitting import MIN_REPLICATE_DOF, compute_weights, replicate_run_variance, residual_run_variance
from .metrics import MetricKind, from_fitting_scale
from .model import DEFAULT_QUANTILES, PredictionError, quantile_suffixes
from .notes import Note
from .schema import SEP
from .uncertainty import DEFAULT_DRAWS, WEBB_WEIGHTS, pseudo_residuals, target_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from .model import ScalingLawModel

#: Runs whose budgets differ by less than this relative amount belong to the same isoFLOP profile.
BUDGET_TOLERANCE = 0.05

#: Profiles the extrapolated line needs before it has a residual of its own.
MIN_BUDGETS = 3

#: A profile whose bootstrap draws lose their minimum more often than this is reported as unstable.
UNSTABLE_FRACTION = 0.05


class IsoflopError(ValueError):
    """Raised when runs cannot be arranged into isoFLOP profiles at all."""


@dataclasses.dataclass(frozen=True)
class IsoflopFit:
    """IsoFLOP profiles of one target and the line through their minima.

    Attributes:
        target: The target column.
        kind: How the target behaves; a target that is better when higher has its *maximum* found.
        model_size: The model-size column.
        dataset_size: The dataset-size column, if the runs had one.
        budgets: The budget of each profile, ascending.
        n_runs: Runs in each profile.
        n_sizes: Distinct model sizes in each profile.
        swept: Smallest and largest model size in each profile, shape ``(n_budgets, 2)``.
        log_optimum: Point estimate of ``log N*`` per profile; NaN where the profile is not convex.
        log_optimum_draws: Bootstrap draws of the same, shape ``(n_draws, n_budgets)``.
        optimum_value_draws: Draws of the target at each profile's minimum, on the fitting scale.
        slope: Point estimate of ``d log N* / d log C``.
        intercept: Point estimate of the line's intercept at ``log C = 0``.
        slope_draws: Draws of the slope.
        intercept_draws: Draws of the intercept.
        cost: The cost model, if one was given; needed to turn ``N*`` into ``D*``.
        notes: Objections raised while fitting.
    """

    target: str
    kind: MetricKind
    model_size: str
    dataset_size: str | None
    budgets: np.ndarray
    n_runs: np.ndarray
    n_sizes: np.ndarray
    swept: np.ndarray
    log_optimum: np.ndarray
    log_optimum_draws: np.ndarray
    optimum_value_draws: np.ndarray
    slope: float
    intercept: float
    slope_draws: np.ndarray
    intercept_draws: np.ndarray
    cost: CostModel | None
    notes: tuple[Note, ...]

    @property
    def warnings(self) -> tuple[Note, ...]:
        """Every objection raised while fitting (an alias of :attr:`notes`, as on fitted models)."""
        return self.notes

    def profiles(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pl.DataFrame:
        """One row per budget: its optimal model size and the target there, with intervals.

        ``bracketed`` says whether the point estimate of the minimum lies inside the model sizes run;
        ``p_not_convex`` is the fraction of draws in which the profile had no minimum at all.
        """
        suffixes = _suffixes(quantiles)
        rows = []
        for j, budget in enumerate(self.budgets):
            draws = np.exp(self.log_optimum_draws[:, j])
            values = from_fitting_scale(self._sign * self.optimum_value_draws[:, j], self.kind)
            point = np.exp(self.log_optimum[j])
            row: dict[str, Any] = {
                "budget": float(budget),
                "n_runs": int(self.n_runs[j]),
                "n_model_sizes": int(self.n_sizes[j]),
                "bracketed": bool(self.swept[j, 0] <= point <= self.swept[j, 1])
                if np.isfinite(point)
                else False,
                "p_not_convex": float(np.mean(~np.isfinite(draws))),
            }
            for column, sample in ((self.model_size, draws), (self.target, values)):
                for q, suffix in zip(quantiles, suffixes, strict=True):
                    row[f"{column}__{suffix}"] = _quantile(sample, q)
            rows.append(row)
        return pl.DataFrame(rows)

    def exponent(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> dict[str, float]:
        """Quantiles of ``a`` in ``N* ∝ C**a``: how fast the optimal model grows with the budget."""
        return {
            s: _quantile(self.slope_draws, q) for q, s in zip(quantiles, _suffixes(quantiles), strict=True)
        }

    def predict(
        self, budgets: Sequence[float], quantiles: Sequence[float] = DEFAULT_QUANTILES
    ) -> pl.DataFrame:
        """The optimal model size -- and, given a cost model, dataset size -- at any budget.

        Budgets outside the range that was profiled are extrapolations of the line, and are labelled so,
        with ``extrapolation_distance`` in units of the profiled range of ``log C``.

        Raises:
            PredictionError: If no line could be fit, or a budget is invalid.
        """
        if not np.isfinite(self.slope):
            raise PredictionError("No line through the profile minima could be fit; see the fit's notes")
        budgets = [float(b) for b in budgets]
        if not budgets or not all(np.isfinite(b) and b > 0 for b in budgets):
            raise PredictionError(f"Budgets must be positive and finite, got {budgets}")
        suffixes = _suffixes(quantiles)
        low, high = np.log(self.budgets.min()), np.log(self.budgets.max())
        span = max(high - low, np.finfo(float).eps)
        rows = []
        for budget in budgets:
            log_c = np.log(budget)
            n_star = np.exp(self.intercept_draws + self.slope_draws * log_c)
            excursion = max(0.0, log_c - high, low - log_c) / span
            row: dict[str, Any] = {
                "budget": budget,
                "domain": "extrapolation" if excursion > 0 else "interpolation",
                "extrapolation_distance": float(excursion),
            }
            samples = [(self.model_size, n_star)]
            if self.cost is not None and self.dataset_size is not None:
                samples.append((self.dataset_size, self.cost.dataset_size_for(budget, n_star)))
            for column, sample in samples:
                for q, suffix in zip(quantiles, suffixes, strict=True):
                    row[f"{column}__{suffix}"] = _quantile(sample, q)
            rows.append(row)
        return pl.DataFrame(rows)

    def compare(
        self,
        model: ScalingLawModel,
        budgets: Sequence[float],
        cost: CostModel | None = None,
        quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    ) -> pl.DataFrame:
        """The two frontiers side by side: these profiles, and the fitted law's.

        ``agree`` is whether the two intervals for ``N*`` overlap. Where they do not, at least one
        method is wrong at that budget -- most often the law, whose shape is being extrapolated -- and
        an ``isoflop_law_disagreement`` warning is raised. Neither answer is chosen for you.

        Args:
            model: A fitted scaling law over the same target.
            budgets: Budgets to compare at.
            cost: The cost law; by default the one this fit was given.
            quantiles: Quantiles to report; the outermost two define the intervals compared.

        Raises:
            PredictionError: If no cost model is available, or the model has no fit for the target.
        """
        cost = cost or self.cost
        if cost is None:
            raise PredictionError("Comparing with a fitted law's frontier needs a cost model")
        if self.target not in model.fits:
            raise PredictionError(f"The model has no fit for {self.target!r}")
        suffixes = _suffixes(quantiles)
        lo, hi = suffixes[int(np.argmin(quantiles))], suffixes[int(np.argmax(quantiles))]
        ours = self.predict(budgets, quantiles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = model.optimal_allocation(budgets, cost, targets=[self.target], quantiles=quantiles)
        column = self.model_size
        frame = pl.DataFrame(
            {
                "budget": ours["budget"],
                **{f"isoflop__{s}": ours[f"{column}__{s}"] for s in suffixes},
                **{f"law__{s}": theirs[f"{column}__{s}"] for s in suffixes},
            }
        ).with_columns(
            (
                (pl.col(f"isoflop__{lo}") <= pl.col(f"law__{hi}"))
                & (pl.col(f"law__{lo}") <= pl.col(f"isoflop__{hi}"))
            ).alias("agree")
        )
        if not frame["agree"].all():
            warnings.warn(
                f"[isoflop_law_disagreement] At {int((~frame['agree']).sum())} budget(s) the isoFLOP "
                "profiles and the fitted law put the optimal model size in non-overlapping intervals. At "
                "least one is wrong there; the law, whose shape is being extrapolated, is the usual suspect.",
                UserWarning,
                stacklevel=2,
            )
        return frame

    @property
    def _sign(self) -> float:
        return -1.0 if self.kind.direction == "higher" else 1.0


def _suffixes(quantiles: Sequence[float]) -> list[str]:
    quantiles = [float(q) for q in quantiles]
    if any(not 0 <= q <= 1 for q in quantiles):
        raise PredictionError(f"quantiles must lie in [0, 1], got {quantiles}")
    try:
        return quantile_suffixes(quantiles)
    except ValueError as error:
        raise PredictionError(str(error)) from None


def _quantile(sample: np.ndarray, q: float) -> float | None:
    finite = sample[np.isfinite(sample)]
    return float(np.quantile(finite, q)) if finite.size else None


def _group_budgets(log_budget: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign each run to a profile: runs within ``tolerance`` of a profile's smallest budget join it.

    Examples:
        >>> groups, centres = _group_budgets(np.log(np.array([1e18, 1.02e18, 1e19, 0.99e19])), 0.05)
        >>> groups.tolist(), [f"{c:.3g}" for c in np.exp(centres)]
        ([0, 0, 1, 1], ['1.01e+18', '9.95e+18'])
    """
    order = np.argsort(log_budget)
    groups = np.empty(log_budget.size, dtype=int)
    group, start = -1, -np.inf
    for i in order:
        if log_budget[i] > start + np.log1p(tolerance):
            group, start = group + 1, log_budget[i]
        groups[i] = group
    centres = np.array([log_budget[groups == g].mean() for g in range(group + 1)])
    return groups, centres


def _run_budgets(
    dataset: Any,
    observations: Any,
    n: np.ndarray,
    d: np.ndarray | None,
    cost: CostModel | None,
    compute: str | None,
) -> np.ndarray:
    """Each run's budget: its recorded cost if a compute column exists, else the cost model's."""
    columns = dataset.frame.columns
    if compute is None:
        found = [c for c in columns if c.startswith(f"{COMPUTE_ROLE}{SEP}")]
        compute = found[0] if len(found) == 1 else None
    if compute is not None:
        if compute not in columns:
            raise IsoflopError(f"Compute column {compute!r} is not in the table")
        run_col = dataset.schema.training_run_id
        recorded = dataset.frame.group_by(run_col).agg(pl.col(compute).cast(pl.Float64, strict=False).mean())
        lookup = dict(zip(recorded[run_col].to_list(), recorded[compute].to_list(), strict=True))
        budgets = np.array([lookup.get(run, np.nan) for run in observations.run_ids], dtype=float)
        if not np.all(np.isfinite(budgets) & (budgets > 0)):
            raise IsoflopError(f"Every run needs a positive recorded cost in {compute!r}")
        return budgets
    if cost is None or d is None:
        raise IsoflopError(
            "Each run's budget comes from a recorded compute__* column or, failing that, from a cost model "
            "applied to its model and dataset size; neither is available"
        )
    return cost(n, d)


def fit_isoflop(
    source: Any,
    cost: CostModel | None = None,
    target: str | None = None,
    columns: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    compute: str | None = None,
    budget_tolerance: float = BUDGET_TOLERANCE,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    degree: int = 3,
) -> IsoflopFit:
    """Fit isoFLOP profiles and the line through their minima.

    Args:
        source: Evaluation records, as for :func:`simple_scaling_laws.fit`, laid out as sweeps of model
            size at a few fixed budgets.
        cost: The cost law. Used to assign budgets when there is no recorded cost, and to turn ``N*``
            into ``D*`` in :meth:`IsoflopFit.predict`.
        target: The target to profile; by default the primary target.
        columns: Optional column-role overrides.
        targets: Optional explicit per-target descriptions.
        compute: The recorded-cost column; by default the single ``compute__*`` column, if any.
        budget_tolerance: Relative difference within which two runs' budgets count as the same.
        n_draws: Bootstrap draws.
        seed: Seed for the bootstrap.
        degree: Degree of the polynomial in ``log N`` fit to each profile: 3 (the default) or 2, the
            classic parabola. See the module docstring for why the default is not the parabola.

    Returns:
        The :class:`IsoflopFit`.

    Raises:
        IsoflopError: If budgets cannot be assigned, ``degree`` is not 2 or 3, or no profile has
            enough model sizes to fit.

    Examples:
        >>> from simple_scaling_laws.cost import cost_model
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> design = [
        ...     (n, float(chinchilla.dataset_size_for(c, n)))
        ...     for c in (1e15, 1e16, 1e17)
        ...     for n in np.sqrt(c / 6.0 / 10.0) * np.logspace(-0.6, 0.6, 5)
        ... ]
        >>> runs = simulate_runs(
        ...     {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.35, "B": 2.0, "beta": 0.35}},
        ...     configurations=design,
        ...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
        ...     run_sd=0.002,
        ...     eval_sd=0.002,
        ...     seed=0,
        ... )
        >>> profiles = fit_isoflop(runs, cost=chinchilla, n_draws=200)
        >>> profiles.profiles()["bracketed"].to_list()
        [True, True, True]
        >>> round(profiles.exponent()["median"], 2)  # alpha = beta, so N* grows as the square root of C
        0.5
    """
    if degree not in (2, 3):
        raise IsoflopError(f"degree must be 2 or 3, got {degree}")
    dataset = build_dataset(source, columns=columns, targets=targets)
    schema = dataset.schema
    if len(schema.model_size) != 1 or len(schema.dataset_size) > 1:
        raise IsoflopError(
            "IsoFLOP profiles are over one model-size column (and at most one dataset-size column), got "
            f"{list(schema.model_size)} and {list(schema.dataset_size)}"
        )
    target = target or schema.primary_target
    if target not in dataset.observations:
        raise DataError(f"Target {target!r} has no usable values")
    observations = dataset.observations[target]
    kind = schema.target(target).kind
    sign = -1.0 if kind.direction == "higher" else 1.0

    config = observations.config_index
    n = dataset.config_values[config, schema.predictors.index(schema.model_size[0])]
    d = (
        dataset.config_values[config, schema.predictors.index(schema.dataset_size[0])]
        if schema.dataset_size
        else None
    )
    groups, centres = _group_budgets(
        np.log(_run_budgets(dataset, observations, n, d, cost, compute)), budget_tolerance
    )
    log_n = np.log(n)
    y = sign * observations.mean
    per_run_var = observations.eval_var_independent / np.maximum(observations.n_eval, 1)

    notes: list[Note] = []
    profiles = []
    for g, centre in enumerate(centres):
        members = np.flatnonzero(groups == g)
        sizes = np.unique(log_n[members]).size
        if sizes < degree + 1:
            notes.append(_note_too_few(np.exp(centre), sizes, degree, dropped=True))
            continue
        if sizes < degree + 2:
            notes.append(_note_too_few(np.exp(centre), sizes, degree, dropped=False))
        profiles.append((centre, members))
    if not profiles:
        raise IsoflopError(
            f"No budget has runs at {degree + 1} or more model sizes, so no profile can be fit"
        )

    def stage_one(weights: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Per profile, the map from run means to quadratic coefficients; fitted values and leverages."""
        maps, fitted, leverage = [], np.zeros_like(y), np.zeros_like(y)
        for _, members in profiles:
            x = log_n[members] - log_n[members].mean()
            design = np.column_stack([x**p for p in range(degree + 1)])
            w = weights[members]
            gram_inverse = np.linalg.inv(design.T @ (w[:, None] * design))
            projection = gram_inverse @ (design * w[:, None]).T  # (3, n_members)
            maps.append(projection)
            fitted[members] = design @ (projection @ y[members])
            leverage[members] = np.einsum("ij,jk,ik->i", design, gram_inverse, design) * w
        return maps, fitted, leverage

    _, fitted, leverage = stage_one(np.ones_like(y))
    n_params = (degree + 1) * len(profiles)
    replicate, replicate_dof = replicate_run_variance(observations, observations.eval_var_independent)
    residual, _ = residual_run_variance(y - fitted, leverage, per_run_var, n_params)
    if replicate is not None and replicate_dof >= MIN_REPLICATE_DOF:
        run_var = replicate
    else:
        run_var = residual if residual is not None else (replicate or 0.0)
    weights = compute_weights(run_var, per_run_var)
    maps, fitted, leverage = stage_one(weights)

    rng = np.random.default_rng(target_seed(seed, target))
    in_profiles = np.concatenate([members for _, members in profiles])
    noise_var = run_var + per_run_var
    if y.size - n_params > 0:
        pseudo = np.zeros_like(y)
        pseudo[in_profiles] = pseudo_residuals(
            (y - fitted)[in_profiles],
            leverage[in_profiles],
            weights[in_profiles],
            n_params,
            config[in_profiles],
            noise_var[in_profiles],
        )
        clusters, cluster_of = np.unique(config, return_inverse=True)
        multipliers = rng.choice(np.asarray(WEBB_WEIGHTS), size=(n_draws, clusters.size))[:, cluster_of]
        synthetic = fitted[None, :] + multipliers * pseudo[None, :]
    else:
        notes.append(
            Note(
                "parametric_uncertainty",
                "warning",
                "Every profile is fit exactly, so uncertainty was simulated from the estimated noise "
                "rather than resampled from residuals.",
                {"target": target},
            )
        )
        synthetic = fitted[None, :] + rng.normal(size=(n_draws, y.size)) * np.sqrt(noise_var)[None, :]

    budgets = np.exp(np.array([centre for centre, _ in profiles]))
    point, _ = _vertices(maps, profiles, y[None, :], log_n)
    draws, draw_values = _vertices(maps, profiles, synthetic, log_n)
    swept = np.array([[np.exp(log_n[m].min()), np.exp(log_n[m].max())] for _, m in profiles])

    for j, budget in enumerate(budgets):
        if not np.isfinite(point[0, j]):
            notes.append(
                Note(
                    "isoflop_profile_not_convex",
                    "warning",
                    f"The profile at budget {budget:.3g} curves the wrong way, so it has no minimum; it is "
                    "left out of the line. Its model sizes probably all sit on one side of the optimum.",
                    {"budget": float(budget)},
                )
            )
        elif not swept[j, 0] <= np.exp(point[0, j]) <= swept[j, 1]:
            notes.append(
                Note(
                    "isoflop_minimum_not_bracketed",
                    "warning",
                    f"The profile at budget {budget:.3g} has its minimum at {np.exp(point[0, j]):.3g}, "
                    f"outside the sizes run ({swept[j, 0]:.3g}-{swept[j, 1]:.3g}): the profile is already "
                    "extrapolating. Run sizes on the other side of it.",
                    {"budget": float(budget), "minimum": float(np.exp(point[0, j]))},
                )
            )
        unstable = float(np.mean(~np.isfinite(draws[:, j])))
        if np.isfinite(point[0, j]) and unstable > UNSTABLE_FRACTION:
            notes.append(
                Note(
                    "isoflop_unstable_minimum",
                    "warning",
                    f"In {unstable:.0%} of bootstrap draws the profile at budget {budget:.3g} had no "
                    "minimum: the valley is too shallow for its noise. Its interval leaves those draws out.",
                    {"budget": float(budget), "fraction": unstable},
                )
            )

    log_c = np.log(budgets)
    slope, intercept = _lines(log_c, point)
    slope_draws, intercept_draws = _lines(log_c, draws)
    usable = int(np.isfinite(point[0]).sum())
    if usable < MIN_BUDGETS:
        notes.append(
            Note(
                "isoflop_too_few_budgets",
                "error" if usable < 2 else "warning",
                f"Only {usable} profile(s) have a minimum, so the line through them "
                + (
                    "cannot be fit."
                    if usable < 2
                    else "has no residual to judge it by and its intervals are unreliable."
                ),
                {"n_budgets": usable, "recommended_minimum": MIN_BUDGETS},
            )
        )

    return IsoflopFit(
        target=target,
        kind=kind,
        model_size=schema.model_size[0],
        dataset_size=schema.dataset_size[0] if schema.dataset_size else None,
        budgets=budgets,
        n_runs=np.array([m.size for _, m in profiles]),
        n_sizes=np.array([np.unique(log_n[m]).size for _, m in profiles]),
        swept=swept,
        log_optimum=point[0],
        log_optimum_draws=draws,
        optimum_value_draws=draw_values,
        slope=float(slope[0]),
        intercept=float(intercept[0]),
        slope_draws=slope_draws,
        intercept_draws=intercept_draws,
        cost=cost,
        notes=tuple(notes),
    )


def _note_too_few(budget: float, sizes: int, degree: int, dropped: bool) -> Note:
    required = degree + 2
    return Note(
        "isoflop_too_few_profile_points",
        "error",
        f"The profile at budget {budget:.3g} has only {sizes} distinct model size(s); a degree-{degree} "
        f"profile needs {required} before it has any residual to be judged by"
        + (
            f", and fewer than {degree + 1} cannot be fit at all, so this budget was left out."
            if dropped
            else "."
        ),
        {"budget": float(budget), "n_sizes": int(sizes), "required": required},
    )


def _vertices(
    maps: list[np.ndarray], profiles: list[tuple[float, np.ndarray]], y: np.ndarray, log_n: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum of each profile's polynomial, per row of ``y``: ``log N*`` and the value there.

    For a parabola, the vertex; for a cubic, the root of the derivative where the curve turns upward. NaN
    where there is no minimum.
    """
    log_star = np.full((y.shape[0], len(profiles)), np.nan)
    value = np.full_like(log_star, np.nan)
    for j, ((_, members), projection) in enumerate(zip(profiles, maps, strict=True)):
        coefficients = y[:, members] @ projection.T  # (rows, degree + 1)
        c1, c2 = coefficients[:, 1], coefficients[:, 2]
        if coefficients.shape[1] == 3:
            convex = c2 > 0
            x = -c1 / (2.0 * np.where(convex, c2, 1.0))
        else:
            c3 = coefficients[:, 3]
            # Roots of c1 + 2 c2 x + 3 c3 x**2; the minimum is the one where 2 c2 + 6 c3 x > 0, which
            # is x = (-c2 + sqrt(c2**2 - 3 c1 c3)) / (3 c3) -- written stably for small c3.
            discriminant = c2**2 - 3.0 * c1 * c3
            root = np.sqrt(np.maximum(discriminant, 0.0))
            denominator = c2 + root
            convex = (discriminant > 0) & (denominator > 0)
            x = -c1 / np.where(convex, denominator, 1.0)
        powers = x[:, None] ** np.arange(coefficients.shape[1])[None, :]
        log_star[:, j] = np.where(convex, x + log_n[members].mean(), np.nan)
        value[:, j] = np.where(convex, (coefficients * powers).sum(axis=1), np.nan)
    return log_star, value


def _lines(log_c: np.ndarray, log_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line of ``log N*`` on ``log C`` per row, over the profiles that have a minimum."""
    mask = np.isfinite(log_star)
    count = mask.sum(axis=1)
    x = np.where(mask, log_c[None, :], 0.0)
    yv = np.where(mask, log_star, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = x.sum(axis=1) / count
        mean_y = yv.sum(axis=1) / count
        dx = np.where(mask, log_c[None, :] - mean_x[:, None], 0.0)
        slope = (dx * (yv - mean_y[:, None])).sum(axis=1) / (dx**2).sum(axis=1)
    slope = np.where(count >= 2, slope, np.nan)
    return slope, mean_y - slope * mean_x
