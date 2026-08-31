"""Design validation, goodness of fit, and cross-metric diagnostics.

Everything here is advisory: the package returns a fit whenever one is mathematically possible and records its
objections rather than raising them, so an automated caller can decide how much to trust a prediction.
Correlations between the primary loss and the other metrics are always computed on **run-level means**;
computing them over raw evaluation rows would count the same trained model dozens of times and inflate both
the estimate and its apparent precision.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from .laws import EXPONENT
from .notes import Note

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from .data import Dataset, TargetObservations
    from .fitting import Bounds, CurveFit, VarianceComponents
    from .laws import LawInstance
    from .uncertainty import Uncertainty

#: Absolute floor on the number of distinct configurations. The effective threshold is the larger
#: of this and ``n_params + 2``: two spare configurations are the minimum that leaves any residual
#: variation to judge the fit by, or to resample in the bootstrap.
MIN_CONFIGURATIONS = 4

#: Distinct levels a predictor needs before its exponent is identified at all. Restricted to one
#: predictor, every law here reduces to ``E + A * x**-alpha``: three free parameters, so two points
#: can always be fit exactly by infinitely many (E, A, alpha) triples.
MIN_PREDICTOR_LEVELS = 3

#: Absolute correlation between log predictors above which they are called collinear.
COLLINEARITY_THRESHOLD = 0.95

#: An exponent is called weakly identified when its 95% interval is wider than this, or wider than
#: the estimate itself. Real scaling exponents are well under one, so an interval half a unit wide
#: is uninformative however large the estimate.
WEAK_IDENTIFICATION_WIDTH = 0.5

#: Ratio of residual-based to replicate-based run variance above which lack of fit is reported.
LACK_OF_FIT_RATIO = 4.0

#: Bootstrap replicates used for correlation confidence intervals.
CORRELATION_DRAWS = 1000

#: Points per predictor on the common grid used to compare targets' fitted curves.
SIMILARITY_GRID_POINTS = 5


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when either series has no variation.

    Written out rather than delegated to ``numpy.corrcoef`` because the bootstrap calls it a thousand times
    per target, where the wrapper overhead dominates the arithmetic.
    """
    if a.size < 3:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    if denominator <= 0:
        return None
    value = float((a @ b) / denominator)
    return value if np.isfinite(value) else None


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Pearson and Spearman correlation, returning ``None`` where undefined."""
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman = float(stats.spearmanr(a, b).statistic)
    return _pearson(a, b), (spearman if np.isfinite(spearman) else None)


def design_notes(dataset: Dataset, law: LawInstance) -> list[Note]:
    """Warnings about whether the experiment can identify the law at all.

    Args:
        dataset: The prepared dataset.
        law: The instantiated law.

    Returns:
        A list of :class:`~simple_scaling_laws.notes.Note`.

    Examples:
        A design with a single dataset size cannot identify its dataset-size exponent:

        >>> import polars as pl
        >>> from .data import build_dataset
        >>> from .laws import build_law
        >>> frame = pl.DataFrame(
        ...     {
        ...         "training_run_id": ["r1", "r2", "r3"],
        ...         "model_size__n": [1e6, 1e7, 1e8],
        ...         "dataset_size__d": [1e4, 1e4, 1e4],
        ...         "test_loss__ce": [2.0, 1.7, 1.5],
        ...     }
        ... )
        >>> dataset = build_dataset(frame)
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> sorted({note.code for note in design_notes(dataset, law)})
        ['no_dataset_size_variation', 'no_repeated_evaluations',
         'single_run_per_configuration', 'too_few_configurations', 'underdetermined']
    """
    notes: list[Note] = []
    n_configs = dataset.n_configurations
    if n_configs < law.n_params:
        notes.append(
            Note(
                "underdetermined",
                "error",
                f"The law has {law.n_params} parameters but the experiment contains only "
                f"{n_configs} distinct scaling configuration(s). The fit cannot be uniquely "
                "determined; predictions are extrapolations of an unidentified curve.",
                {"n_configurations": n_configs, "n_params": law.n_params},
            )
        )
    threshold = max(MIN_CONFIGURATIONS, law.n_params + 2)
    if n_configs < threshold:
        notes.append(
            Note(
                "too_few_configurations",
                "warning",
                f"Only {n_configs} distinct scaling configuration(s) for a {law.n_params}-parameter "
                f"law; {threshold} is the minimum that leaves enough spare variation to judge the "
                "fit or to resample it. Exponents estimated from so few scales are unreliable, and "
                "adding configurations helps far more than adding runs or evaluations.",
                {"n_configurations": n_configs, "recommended_minimum": threshold},
            )
        )

    for column in law.predictors:
        j = dataset.schema.predictors.index(column)
        levels = int(np.unique(dataset.config_values[:, j]).size)
        if 2 <= levels < MIN_PREDICTOR_LEVELS:
            notes.append(
                Note(
                    "too_few_predictor_levels",
                    "error",
                    f"Predictor {column!r} was measured at only {levels} distinct values. Its "
                    "exponent is not identified: two points can be fit exactly by any number of "
                    f"different curves, so the fitted exponent, amplitude and offset are arbitrary "
                    "and their intervals do not cover the truth. Add a third scale for this "
                    "predictor.",
                    {"predictor": column, "n_levels": levels, "required": MIN_PREDICTOR_LEVELS},
                )
            )

    for role, columns in (
        ("model_size", dataset.schema.model_size),
        ("dataset_size", dataset.schema.dataset_size),
    ):
        for column in columns:
            j = dataset.schema.predictors.index(column)
            if np.unique(dataset.config_values[:, j]).size < 2:
                notes.append(
                    Note(
                        f"no_{role}_variation",
                        "warning",
                        f"Predictor {column!r} takes a single value, so its exponent carries no "
                        "information and is absorbed into the offset.",
                        {"predictor": column},
                    )
                )

    log_values = np.log(dataset.config_values)
    if log_values.shape[0] >= 3:
        for i in range(log_values.shape[1]):
            for j in range(i + 1, log_values.shape[1]):
                if np.ptp(log_values[:, i]) == 0 or np.ptp(log_values[:, j]) == 0:
                    continue
                correlation = float(np.corrcoef(log_values[:, i], log_values[:, j])[0, 1])
                if abs(correlation) > COLLINEARITY_THRESHOLD:
                    notes.append(
                        Note(
                            "collinear_predictors",
                            "warning",
                            f"Predictors {dataset.schema.predictors[i]!r} and "
                            f"{dataset.schema.predictors[j]!r} are collinear in log space "
                            f"(r = {correlation:.3f}), so their separate exponents cannot be "
                            "distinguished from each other.",
                            {
                                "predictors": [
                                    dataset.schema.predictors[i],
                                    dataset.schema.predictors[j],
                                ],
                                "log_correlation": correlation,
                            },
                        )
                    )

    if int(dataset.runs_per_config.max()) < 2:
        notes.append(
            Note(
                "single_run_per_configuration",
                "warning",
                "No scaling configuration has more than one training run, so run-to-run training "
                "variation cannot be separated from the law's own lack of fit. The run-level "
                "variance falls back to a residual-based estimate, which is biased upward.",
                {},
            )
        )
    primary = dataset.observations[dataset.schema.primary_target]
    if int(primary.n_eval.max()) < 2:
        notes.append(
            Note(
                "no_repeated_evaluations",
                "warning",
                "No training run has more than one evaluation row, so finite-evaluation-set noise "
                "cannot be estimated separately and is absorbed into the run-level variance.",
                {},
            )
        )
    return notes


def fit_notes(
    target: str,
    law: LawInstance,
    fit: CurveFit,
    bounds: Bounds,
    uncertainty: Uncertainty,
    components: VarianceComponents,
) -> list[Note]:
    """Warnings about one target's fitted curve.

    Args:
        target: The target column name.
        law: The instantiated law.
        fit: The point-estimate fit.
        bounds: The parameter box constraints used.
        uncertainty: The bootstrap draws.
        components: The estimated variance components.

    Returns:
        A list of :class:`~simple_scaling_laws.notes.Note`.
    """
    notes: list[Note] = []
    if not fit.success:
        notes.append(
            Note(
                "optimizer_nonconvergence",
                "warning",
                f"The optimizer did not report convergence for {target!r}: {fit.message}",
                {"target": target, "message": fit.message},
            )
        )
    pinned = bounds.at_bound(fit.params, law.param_names)
    if pinned:
        notes.append(
            Note(
                "parameter_at_bound",
                "warning",
                f"Parameter(s) {list(pinned)} for {target!r} sit on a constraint boundary. Their "
                "point estimates and intervals should not be read as interior optima.",
                {"target": target, "parameters": list(pinned)},
            )
        )

    lower, upper = uncertainty.quantiles([0.025, 0.975])
    for i, (name, kind) in enumerate(zip(law.param_names, law.param_kinds, strict=True)):
        if kind != EXPONENT:
            continue
        width = float(upper[i] - lower[i])
        estimate = float(fit.params[i])
        if width > max(WEAK_IDENTIFICATION_WIDTH, abs(estimate)):
            notes.append(
                Note(
                    "weakly_identified_exponent",
                    "warning",
                    f"Exponent {name!r} for {target!r} is {estimate:.3g} with a 95% interval of "
                    f"[{lower[i]:.3g}, {upper[i]:.3g}] -- wider than the estimate itself. The data "
                    "do not pin down how fast this predictor improves performance, so predictions "
                    "at new scales of it are little more than a guess.",
                    {
                        "target": target,
                        "parameter": name,
                        "estimate": estimate,
                        "interval": [lower[i], upper[i]],
                    },
                )
            )

    replicate, residual = components.run_var_replicate, components.run_var_residual
    if replicate is not None and residual is not None and replicate > 0:
        ratio = residual / replicate
        if ratio > LACK_OF_FIT_RATIO:
            notes.append(
                Note(
                    "lack_of_fit",
                    "warning",
                    f"For {target!r} the scatter around the fitted curve is {ratio:.1f}x larger than "
                    "the scatter between repeated training runs at the same scale. The law's "
                    "functional form does not describe these data well.",
                    {"target": target, "ratio": float(ratio)},
                )
            )
    return notes


def goodness_of_fit(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray, n_params: int) -> dict[str, Any]:
    """Summary fit quality on run-level means.

    Args:
        y: Run-level means.
        fitted: Fitted values at the same points.
        weights: Fitting weights.
        n_params: Number of fitted parameters.

    Returns:
        A dictionary of goodness-of-fit statistics.

    Examples:
        >>> gof = goodness_of_fit(
        ...     np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.ones(3), 2
        ... )
        >>> gof["r_squared"], gof["rmse"]
        (1.0, 0.0)
    """
    residual = y - fitted
    weighted_mean = float(np.average(y, weights=weights))
    ss_residual = float(np.sum(weights * residual**2))
    ss_total = float(np.sum(weights * (y - weighted_mean) ** 2))
    return {
        "n_observations": int(y.size),
        "n_params": int(n_params),
        "residual_dof": int(y.size - n_params),
        "r_squared": (1.0 - ss_residual / ss_total) if ss_total > 0 else None,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "weighted_rmse": float(np.sqrt(ss_residual / y.size)),
        "max_abs_residual": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def _aligned_run_means(
    primary: TargetObservations, other: TargetObservations
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run means of two targets restricted to the runs both observed, with configuration indices."""
    shared = [run for run in primary.run_ids if run in set(other.run_ids)]
    primary_position = {run: i for i, run in enumerate(primary.run_ids)}
    other_position = {run: i for i, run in enumerate(other.run_ids)}
    a = np.array([primary.mean[primary_position[r]] for r in shared])
    b = np.array([other.mean[other_position[r]] for r in shared])
    configs = np.array([primary.config_index[primary_position[r]] for r in shared], dtype=int)
    return a, b, configs


def metric_correlations(dataset: Dataset, seed: int = 0, n_draws: int = CORRELATION_DRAWS) -> dict[str, Any]:
    """Associations between the primary target and every other target.

    Correlations are computed across **training runs**, using each run's mean, so a model evaluated
    on a hundred bootstrap resamples counts once. The correlation over raw evaluation rows is
    reported alongside purely so the inflation is visible; it is not a valid estimate of how the
    metrics co-vary across trained models.

    Confidence intervals come from resampling scaling configurations with replacement, which
    respects the fact that runs at the same scale are not independent draws.

    Args:
        dataset: The prepared dataset.
        seed: Seed for the bootstrap.
        n_draws: Number of bootstrap replicates.

    Returns:
        A mapping from target name to its correlation diagnostics.
    """
    rng = np.random.default_rng(seed)
    primary_name = dataset.schema.primary_target
    primary = dataset.observations[primary_name]
    out: dict[str, Any] = {}
    for name, other in dataset.observations.items():
        if name == primary_name:
            continue
        a, b, configs = _aligned_run_means(primary, other)
        pearson, spearman = _safe_correlation(a, b)
        rows = dataset.frame.select(primary_name, name).drop_nulls()
        row_pearson, row_spearman = _safe_correlation(
            rows[primary_name].to_numpy().astype(float), rows[name].to_numpy().astype(float)
        )
        interval: list[float] | None = None
        if pearson is not None:
            members = [np.flatnonzero(configs == c) for c in np.unique(configs)]
            replicates = []
            for _ in range(n_draws):
                chosen = rng.integers(0, len(members), size=len(members))
                index = np.concatenate([members[c] for c in chosen])
                value = _pearson(a[index], b[index])
                if value is not None:
                    replicates.append(value)
            if replicates:
                interval = [float(q) for q in np.quantile(replicates, [0.025, 0.975])]
        out[name] = {
            "n_runs": int(a.size),
            "pearson": pearson,
            "spearman": spearman,
            "pearson_ci": interval,
            "evaluation_row_pearson": row_pearson,
            "evaluation_row_spearman": row_spearman,
            "n_evaluation_rows": int(rows.height),
        }
    return out


def similarity_grid(
    dataset: Dataset, predictors: Sequence[str], n_points: int = SIMILARITY_GRID_POINTS
) -> np.ndarray:
    """A log-spaced grid spanning the observed range of the given predictors.

    Args:
        dataset: The prepared dataset.
        predictors: The predictor columns to vary, normally the fitted law's own.
        n_points: Points per predictor.

    Returns:
        Log-normalized predictor values, shape ``(n_points ** len(predictors), len(predictors))``.
    """
    indices = dataset.column_indices(predictors)
    axes = []
    for j in indices:
        column = np.log(dataset.config_values[:, j])
        axes.append(np.linspace(column.min(), column.max(), n_points))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh]) - np.log(dataset.reference[indices])


def scaling_similarity(
    dataset: Dataset,
    law: LawInstance,
    params: Mapping[str, np.ndarray],
    targets: Sequence[str],
) -> dict[str, Any]:
    """How similarly each target's fitted curve behaves across the observed domain.

    Metrics have incompatible units and directions, so comparing raw parameters is meaningless.
    Instead each fitted curve is evaluated on a common grid spanning the observed scales, and the
    resulting prediction vectors are correlated with the primary target's. A value near ``-1`` for
    an increasing metric such as AUROC means it improves in lockstep with the loss it is paired
    against.

    Args:
        dataset: The prepared dataset.
        law: The instantiated law.
        params: Fitted parameter vectors, keyed by target.
        targets: Targets to compare.

    Returns:
        A mapping from target name to its similarity diagnostics.
    """
    grid = similarity_grid(dataset, law.predictors)
    primary_name = dataset.schema.primary_target
    reference = law.evaluate(params[primary_name], grid)
    out: dict[str, Any] = {}
    for name in targets:
        if name == primary_name:
            continue
        predicted = law.evaluate(params[name], grid)
        pearson, spearman = _safe_correlation(reference, predicted)
        out[name] = {
            "grid_pearson": pearson,
            "grid_spearman": spearman,
            "n_grid_points": int(grid.shape[0]),
        }
    return out
