"""Input reading, validation, and reduction of evaluation rows to run-level observations.

The central statistical move of this package happens here: many bootstrap evaluation rows for one trained
model are reduced to **one** observation (that run's mean) plus a count. That is what stops repeated
evaluations of the same model from masquerading as independent training evidence. The spread *within* a run
estimates evaluation noise; the spread *between* runs at the same scale estimates training-run noise.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .metrics import MetricError, to_fitting_scale
from .notes import Note
from .schema import Schema, discover_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

#: Minimum number of shared evaluation resamples before a run pair contributes to the paired
#: evaluation-covariance estimate.
MIN_SHARED_EVALUATIONS = 3


class DataError(ValueError):
    """Raised when an input table cannot be used for scaling-law fitting."""


def read_table(source: str | Path | pl.DataFrame | Any) -> pl.DataFrame:
    """Read an input table from a path or an in-memory dataframe.

    Parquet is the intended interchange format; CSV is accepted for convenience. In-memory Polars
    frames are used as-is and anything else (a pandas frame, a PyArrow table) is converted through
    the dataframe interchange protocol, so callers are not forced to adopt Polars.

    Args:
        source: A ``.parquet``/``.csv`` path, or an in-memory dataframe.

    Returns:
        The table as a Polars ``DataFrame``.

    Raises:
        DataError: If the path suffix is unrecognized or the object cannot be converted.

    Examples:
        >>> frame = pl.DataFrame({"a": [1, 2]})
        >>> read_table(frame).shape
        (2, 1)
        >>> read_table("runs.txt")
        Traceback (most recent call last):
        ...
        simple_scaling_laws.data.DataError: Unsupported input suffix '.txt'; use .parquet or .csv
    """
    if isinstance(source, pl.DataFrame):
        return source
    if isinstance(source, pl.LazyFrame):
        return source.collect()
    if isinstance(source, str | Path):
        path = Path(source)
        match path.suffix.lower():
            case ".parquet" | ".pq":
                return pl.read_parquet(path)
            case ".csv":
                return pl.read_csv(path)
            case other:
                raise DataError(f"Unsupported input suffix {other!r}; use .parquet or .csv")
    try:
        return pl.from_dataframe(source)
    except Exception as exc:  # pragma: no cover - depends on optional third-party frames
        raise DataError(f"Cannot interpret {type(source).__name__} as a dataframe: {exc}") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class TargetObservations:
    """Run-level observations for one target column.

    Attributes:
        target: The target column name.
        run_ids: Identifiers of the runs that have at least one usable row for this target.
        config_index: For each run, the index of its scaling configuration.
        mean: For each run, the mean of its evaluation rows.
        n_eval: For each run, how many evaluation rows that mean averages.
        within_ss: Total within-run sum of squared deviations, pooled over runs.
        within_dof: Degrees of freedom behind ``within_ss`` (``sum_r (n_r - 1)``).
        eval_pair_correlation: Correlation between the evaluation residuals of different runs on
            shared ``test_set_id`` values, or ``None`` if no run pair shares enough resamples.
        n_shared_pairs: Number of run pairs that contributed to ``eval_pair_correlation``.
    """

    target: str
    run_ids: tuple[str, ...]
    config_index: np.ndarray
    mean: np.ndarray
    n_eval: np.ndarray
    within_ss: float
    within_dof: int
    eval_pair_correlation: float | None
    n_shared_pairs: int

    @property
    def n_runs(self) -> int:
        """Number of runs contributing to this target."""
        return len(self.run_ids)

    @property
    def n_configurations(self) -> int:
        """Number of distinct configurations represented among those runs."""
        return int(np.unique(self.config_index).size)

    @property
    def eval_var(self) -> float:
        """Pooled within-run evaluation variance, or 0.0 when no run has repeated evaluations."""
        return float(self.within_ss / self.within_dof) if self.within_dof > 0 else 0.0

    @property
    def eval_var_independent(self) -> float:
        """The part of the evaluation variance that averages away across runs.

        When the same ``test_set_id`` resamples are reused across trained models, part of the evaluation noise
        is *shared*: it shifts every run's mean in the same direction instead of cancelling. Only the
        remaining, independent part is reduced by comparing runs, so only that part belongs in the fitting
        weights.
        """
        rho = 0.0 if self.eval_pair_correlation is None else max(0.0, self.eval_pair_correlation)
        return self.eval_var * (1.0 - rho)


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    """A validated input table reduced to the quantities the fitter needs.

    Attributes:
        schema: The resolved column roles.
        run_ids: All training-run identifiers, sorted.
        run_config_index: For each run, the index of its scaling configuration.
        config_values: Raw predictor values per configuration, shape ``(n_configs, n_predictors)``.
        reference: Per-predictor normalization constant (the geometric mean over configurations).
        observations: Run-level observations, keyed by target column name.
        n_rows: Number of evaluation rows in the input table.
        n_test_sets: Number of distinct ``test_set_id`` values, or ``None`` if the column is absent.
        notes: Validation warnings raised while building.
        frame: The validated input table.
    """

    schema: Schema
    run_ids: tuple[str, ...]
    run_config_index: np.ndarray
    config_values: np.ndarray
    reference: np.ndarray
    observations: dict[str, TargetObservations]
    n_rows: int
    n_test_sets: int | None
    notes: tuple[Note, ...]
    frame: pl.DataFrame = dataclasses.field(repr=False)

    @property
    def n_runs(self) -> int:
        """Number of independently trained models."""
        return len(self.run_ids)

    @property
    def n_configurations(self) -> int:
        """Number of distinct scaling configurations."""
        return int(self.config_values.shape[0])

    @property
    def log_x(self) -> np.ndarray:
        """Log of the normalized predictors, per configuration, shape ``(n_configs, n_predictors)``."""
        return np.log(self.config_values) - np.log(self.reference)

    @property
    def runs_per_config(self) -> np.ndarray:
        """Number of training runs at each configuration."""
        return np.bincount(self.run_config_index, minlength=self.n_configurations)

    def column_indices(self, names: Sequence[str]) -> np.ndarray:
        """Positions of the named predictor columns within :attr:`config_values`.

        Args:
            names: Predictor column names.

        Returns:
            Integer indices, one per name.
        """
        return np.array([self.schema.predictors.index(name) for name in names], dtype=int)

    def varying_predictors(self) -> tuple[str, ...]:
        """Predictor columns that take more than one value across configurations.

        A predictor held fixed for the whole experiment contributes nothing but a constant, which
        is indistinguishable from the law's offset. Keeping such a term would leave the fit
        rank-deficient: its amplitude and exponent could take any values at all without changing a
        single prediction.
        """
        return tuple(
            name
            for j, name in enumerate(self.schema.predictors)
            if np.unique(self.config_values[:, j]).size > 1
        )

    def observed_domain(self) -> dict[str, dict[str, float]]:
        """The observed range of each predictor.

        Returns:
            A mapping from predictor column name to its ``min``, ``max`` and ``reference`` values.
        """
        return {
            name: {
                "min": float(self.config_values[:, j].min()),
                "max": float(self.config_values[:, j].max()),
                "reference": float(self.reference[j]),
            }
            for j, name in enumerate(self.schema.predictors)
        }

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Convert raw predictor values to the log-normalized scale used by the laws.

        Args:
            values: Raw predictor values, shape ``(n_points, n_predictors)``.

        Returns:
            ``log(values / reference)``, same shape.

        Raises:
            DataError: If any value is not strictly positive and finite.
        """
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise DataError("Predictor values must be finite and strictly positive")
        return np.log(values) - np.log(self.reference)


def _validate_predictors(frame: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    """Cast predictors to floats and check they are positive, finite, and constant within a run."""
    frame = frame.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in schema.predictors)
    for column in schema.predictors:
        values = frame[column]
        if values.null_count() or not values.is_finite().all():
            raise DataError(f"Predictor column {column!r} contains null or non-finite values")
        if (values <= 0).any():
            raise DataError(f"Predictor column {column!r} must be strictly positive (it is log-scaled)")
    varying = (
        frame.group_by(schema.training_run_id)
        .agg(pl.col(c).n_unique().alias(c) for c in schema.predictors)
        .filter(pl.any_horizontal(pl.col(c) > 1 for c in schema.predictors))
    )
    if varying.height:
        offenders = varying[schema.training_run_id].to_list()[:5]
        raise DataError(
            f"Predictor values vary within training run(s) {offenders}. Every row of one "
            "training_run_id must describe the same trained model, so its model size and dataset "
            "size must be constant."
        )
    return frame


def _pair_correlation(residuals: np.ndarray) -> tuple[float | None, int]:
    """Mean correlation between run residual vectors over shared evaluation resamples.

    Args:
        residuals: Shape ``(n_runs, n_test_sets)`` with ``NaN`` where a run was not evaluated on a
            resample. Each row is already centered on that run's mean.

    Returns:
        The evaluation-count-weighted mean correlation and the number of contributing run pairs.
    """
    correlations: list[float] = []
    weights: list[float] = []
    for i, j in itertools.combinations(range(residuals.shape[0]), 2):
        shared = np.isfinite(residuals[i]) & np.isfinite(residuals[j])
        n_shared = int(shared.sum())
        if n_shared < MIN_SHARED_EVALUATIONS:
            continue
        a, b = residuals[i, shared], residuals[j, shared]
        a, b = a - a.mean(), b - b.mean()
        denom = float(np.sqrt((a @ a) * (b @ b)))
        if denom <= 0:
            continue
        correlations.append(float((a @ b) / denom))
        weights.append(float(n_shared))
    if not correlations:
        return None, 0
    return float(np.average(correlations, weights=weights)), len(correlations)


def _target_observations(
    frame: pl.DataFrame,
    schema: Schema,
    target: str,
    run_index: Mapping[str, int],
    run_config_index: np.ndarray,
) -> tuple[TargetObservations | None, list[Note]]:
    """Reduce one target's evaluation rows to run-level observations."""
    notes: list[Note] = []
    run_col = schema.training_run_id
    valid = frame.select(
        pl.col(run_col).cast(pl.String),
        pl.col(target).cast(pl.Float64, strict=False).alias("_y"),
        *([pl.col(schema.test_set_id).cast(pl.String).alias("_e")] if schema.test_set_id else []),
    ).filter(pl.col("_y").is_finite())
    kind = schema.target(target).kind
    if kind.transforms and valid.height:
        # Fit on the transformed scale, and transform *before* aggregating: the law is additive
        # there, so a run's summary must be the mean of its transformed evaluation rows rather than
        # the transform of their mean.
        raw = valid["_y"].to_numpy()
        try:
            valid = valid.with_columns(pl.Series("_y", to_fitting_scale(raw, kind)))
        except MetricError as exc:
            raise DataError(f"Target {target!r}: {exc}") from exc
        n_saturated = int(np.sum((raw <= 0.0) | (raw >= 1.0)))
        if n_saturated:
            notes.append(
                Note(
                    "saturated_values",
                    "warning",
                    f"{n_saturated} value(s) of {target!r} sit exactly at 0 or 1, which have no "
                    "finite logit and were nudged just inside the interval. A metric pinned at its "
                    "ceiling usually means the evaluation set is too small to separate models.",
                    {"target": target, "n_saturated": n_saturated},
                )
            )

    n_dropped = frame.height - valid.height
    if n_dropped:
        notes.append(
            Note(
                "dropped_rows",
                "info",
                f"Dropped {n_dropped} row(s) with missing or non-finite {target!r} values.",
                {"target": target, "n_dropped": n_dropped},
            )
        )
    if valid.height == 0:
        notes.append(
            Note("empty_target", "warning", f"Target {target!r} has no usable values; skipping it.", {})
        )
        return None, notes

    grouped = (
        valid.group_by(run_col)
        .agg(
            pl.col("_y").mean().alias("mean"),
            pl.len().alias("n_eval"),
            ((pl.col("_y") - pl.col("_y").mean()) ** 2).sum().alias("ss"),
        )
        .sort(run_col)
    )
    run_ids = tuple(grouped[run_col].to_list())
    positions = np.array([run_index[r] for r in run_ids], dtype=int)
    n_eval = grouped["n_eval"].to_numpy().astype(int)

    rho, n_pairs = None, 0
    if schema.test_set_id is not None and len(run_ids) > 1:
        wide = (
            valid.with_columns((pl.col("_y") - pl.col("_y").mean().over(run_col)).alias("_r"))
            .group_by([run_col, "_e"])
            .agg(pl.col("_r").mean())
            .pivot(on="_e", index=run_col, values="_r")
            .sort(run_col)
        )
        rho, n_pairs = _pair_correlation(wide.drop(run_col).to_numpy().astype(float))

    return (
        TargetObservations(
            target=target,
            run_ids=run_ids,
            config_index=run_config_index[positions],
            mean=grouped["mean"].to_numpy().astype(float),
            n_eval=n_eval,
            within_ss=float(grouped["ss"].sum()),
            within_dof=int(n_eval.sum() - len(run_ids)),
            eval_pair_correlation=rho,
            n_shared_pairs=n_pairs,
        ),
        notes,
    )


def _id_consistency_notes(frame: pl.DataFrame, schema: Schema) -> list[Note]:
    """Warn when identifier columns contradict the documented data model."""
    notes: list[Note] = []
    run_col = schema.training_run_id
    for column, code in (
        (schema.train_set_id, "run_spans_train_sets"),
        (schema.optimizer_seed, "run_spans_optimizer_seeds"),
    ):
        if column is None:
            continue
        bad = (
            frame.group_by(run_col)
            .agg(pl.col(column).n_unique().alias("n"))
            .filter(pl.col("n") > 1)[run_col]
            .to_list()
        )
        if bad:
            notes.append(
                Note(
                    code,
                    "warning",
                    f"Training run(s) {bad[:5]} span multiple {column!r} values. One training_run_id "
                    "should identify one independently trained model.",
                    {"column": column, "n_runs": len(bad)},
                )
            )
    if schema.test_set_id is not None:
        duplicates = (
            frame.group_by([run_col, schema.test_set_id])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        )
        if duplicates:
            notes.append(
                Note(
                    "duplicate_evaluations",
                    "warning",
                    f"{duplicates} (training_run_id, {schema.test_set_id}) combination(s) appear more "
                    "than once. Repeated identical evaluations are treated as independent rows.",
                    {"n_duplicated_pairs": duplicates},
                )
            )
    return notes


def build_dataset(
    source: str | Path | pl.DataFrame | Any,
    columns: Mapping[str, str | Iterable[str]] | None = None,
    primary_target: str | None = None,
    targets: Mapping[str, Any] | None = None,
) -> Dataset:
    """Read, validate, and reduce an input table to run-level observations.

    Args:
        source: A parquet/CSV path or an in-memory dataframe.
        columns: Optional column-role overrides, see :func:`simple_scaling_laws.schema.discover_schema`.
        primary_target: Optional explicit primary target column.
        targets: Optional explicit per-target descriptions, see
            :func:`simple_scaling_laws.schema.discover_schema`.

    Returns:
        The prepared :class:`Dataset`.

    Raises:
        DataError: If the table is empty, or predictors are invalid, or no target is usable.

    Examples:
        >>> frame = pl.DataFrame(
        ...     {
        ...         "training_run_id": ["r1", "r1", "r2", "r2"],
        ...         "test_set_id": ["b1", "b2", "b1", "b2"],
        ...         "model_size__n": [1e6, 1e6, 1e7, 1e7],
        ...         "dataset_size__d": [1e4, 1e4, 1e4, 1e4],
        ...         "test_loss__ce": [2.0, 2.2, 1.6, 1.8],
        ...     }
        ... )
        >>> dataset = build_dataset(frame)
        >>> dataset.n_runs, dataset.n_configurations, dataset.n_rows
        (2, 2, 4)

        Each run contributes one observation -- its mean -- not one per evaluation row:

        >>> observations = dataset.observations["test_loss__ce"]
        >>> observations.mean
        array([2.1, 1.7])
        >>> observations.n_eval
        array([2, 2])

        The spread within each run estimates evaluation noise:

        >>> round(observations.eval_var, 4)
        0.02

        Predictors are normalized by their geometric mean across configurations:

        >>> dataset.reference.round(2)
        array([3162277.66,   10000.  ])
    """
    frame = read_table(source)
    if frame.height == 0:
        raise DataError("Input table is empty")
    schema = discover_schema(frame.columns, columns, primary_target, targets)

    frame = _validate_predictors(frame, schema)
    frame = frame.with_columns(pl.col(schema.training_run_id).cast(pl.String))
    notes = _id_consistency_notes(frame, schema)

    run_table = frame.select(schema.training_run_id, *schema.predictors).unique().sort(schema.training_run_id)
    run_ids = tuple(run_table[schema.training_run_id].to_list())
    run_index = {run: i for i, run in enumerate(run_ids)}
    run_values = run_table.select(schema.predictors).to_numpy().astype(float)

    config_values, run_config_index = np.unique(run_values, axis=0, return_inverse=True)
    run_config_index = np.asarray(run_config_index, dtype=int).reshape(-1)
    reference = np.exp(np.log(config_values).mean(axis=0))

    observations: dict[str, TargetObservations] = {}
    for target in schema.target_names:
        target_observations, target_notes = _target_observations(
            frame, schema, target, run_index, run_config_index
        )
        notes.extend(target_notes)
        if target_observations is not None:
            observations[target] = target_observations
    if not observations:
        raise DataError("No target column has any usable values")
    if schema.primary_target not in observations:
        raise DataError(f"Primary target {schema.primary_target!r} has no usable values")

    return Dataset(
        schema=schema,
        run_ids=run_ids,
        run_config_index=run_config_index,
        config_values=config_values,
        reference=reference,
        observations=observations,
        n_rows=frame.height,
        n_test_sets=(frame[schema.test_set_id].n_unique() if schema.test_set_id else None),
        notes=tuple(notes),
        frame=frame,
    )


def config_matrix(dataset: Dataset, runs: Sequence[int] | np.ndarray) -> np.ndarray:
    """Log-normalized predictors for a sequence of configuration indices.

    Args:
        dataset: The dataset supplying the configurations.
        runs: Configuration indices, typically ``observations.config_index``.

    Returns:
        Array of shape ``(len(runs), n_predictors)``.
    """
    return dataset.log_x[np.asarray(runs, dtype=int)]
