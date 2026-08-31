"""The fitted model: inspection, prediction, and round-tripping through a ``.slaw`` artifact."""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from . import artifact as artifact_io
from .data import DataError, read_table
from .laws import EXPONENT, LawInstance
from .metrics import MetricKind, from_fitting_scale, scale_description
from .notes import Note
from .schema import Schema
from .uncertainty import Uncertainty, from_arrays, to_arrays

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from pathlib import Path

#: Prediction quantiles reported when the caller does not choose.
DEFAULT_QUANTILES: tuple[float, ...] = (0.025, 0.5, 0.975)

#: Supported prediction semantics.
PREDICTION_KINDS: tuple[str, ...] = ("mean", "new-run")

#: Minimum number of observed run-level deviations before training-run noise is resampled from them
#: rather than drawn from a normal distribution.
MIN_DEVIATIONS_FOR_RESAMPLING = 10

#: Column name used for the caller's point identifiers.
POINT_ID = "point_id"


class PredictionError(ValueError):
    """Raised when a prediction request cannot be satisfied."""


#: Quantiles whose column suffix is a word rather than a number. The three endpoints need names
#: because the numeric rule reads the fractional part, where 0.0 and 1.0 are indistinguishable.
NAMED_QUANTILES: dict[float, str] = {0.0: "min", 0.5: "median", 1.0: "max"}


def quantile_suffix(q: float) -> str:
    """Column suffix for a quantile.

    The three endpoints are named in words -- ``min``, ``median`` and ``max``. Every other quantile
    is ``q`` followed by the three digits of its fractional part, so ``0.025`` becomes ``q025`` and
    ``0.1`` becomes ``q100`` (read as a thousandth, i.e. 0.100). Quantiles are distinguished to
    three decimal places.

    Args:
        q: A quantile in ``[0, 1]``.

    Returns:
        The suffix.

    Examples:
        >>> [quantile_suffix(q) for q in (0.025, 0.5, 0.975, 0.1)]
        ['q025', 'median', 'q975', 'q100']

        The endpoints would otherwise collide, since both have a fractional part of ``000``:

        >>> [quantile_suffix(q) for q in (0.0, 1.0)]
        ['min', 'max']
    """
    for value, name in NAMED_QUANTILES.items():
        if abs(q - value) < 5e-4:
            return name
    return "q" + f"{q:.3f}".split(".")[1]


@dataclasses.dataclass(frozen=True, slots=True)
class TargetFit:
    """The fitted scaling law for one target.

    Attributes:
        target: The target column name.
        role: The target's role, e.g. ``"test_loss"``.
        signed_amplitude: Whether the amplitudes were allowed to be negative.
        params: The point-estimate parameter vector.
        variance_components: The estimated variance components.
        optimizer: Optimizer diagnostics.
        goodness_of_fit: Fit-quality statistics.
        uncertainty_method: How the draws were produced.
        n_failed_draws: Bootstrap refits that did not report convergence.
        pairing: What the draws are comparable to. Two fits whose ``pairing`` matches for a target
            used identical cluster multipliers, so differencing their draws measures the variance of
            the difference rather than the sum of two variances.
    """

    target: str
    role: str
    signed_amplitude: bool
    params: np.ndarray
    variance_components: dict[str, Any]
    optimizer: dict[str, Any]
    goodness_of_fit: dict[str, Any]
    uncertainty_method: str
    n_failed_draws: int
    pairing: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def run_sd(self) -> float:
        """The run-to-run training standard deviation used for ``new-run`` predictions."""
        return float(np.sqrt(max(0.0, self.variance_components["run_var"])))

    def to_dict(self, law: LawInstance, reference: np.ndarray) -> dict[str, Any]:
        """Serialize for ``fits.json``.

        Args:
            law: The instantiated law, supplying parameter names.
            reference: The predictor normalization constants, used to report raw-scale amplitudes.

        Returns:
            A JSON-compatible dictionary.
        """
        return {
            "role": self.role,
            "signed_amplitude": self.signed_amplitude,
            "params": {name: float(v) for name, v in zip(law.param_names, self.params, strict=True)},
            "params_raw_scale": raw_scale_params(law, self.params, reference),
            "variance_components": self.variance_components,
            "optimizer": self.optimizer,
            "goodness_of_fit": self.goodness_of_fit,
            "uncertainty": {
                "method": self.uncertainty_method,
                "n_failed_draws": self.n_failed_draws,
                "pairing": self.pairing,
            },
        }

    @classmethod
    def from_dict(cls, target: str, data: Mapping[str, Any], law: LawInstance) -> TargetFit:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            target=target,
            role=data["role"],
            signed_amplitude=bool(data["signed_amplitude"]),
            params=np.array([float(data["params"][name]) for name in law.param_names]),
            variance_components=dict(data["variance_components"]),
            optimizer=dict(data["optimizer"]),
            goodness_of_fit=dict(data["goodness_of_fit"]),
            uncertainty_method=data["uncertainty"]["method"],
            n_failed_draws=int(data["uncertainty"]["n_failed_draws"]),
            pairing=dict(data["uncertainty"].get("pairing") or {}),
        )


def raw_scale_params(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Parameters restated in raw predictor units.

    Fitting happens on normalized predictors ``x / reference``, which keeps amplitudes on the
    target's own scale and the optimization well conditioned. Exponents are unaffected by that
    choice, but amplitudes are: an amplitude ``A`` on normalized predictors corresponds to
    ``A * reference ** alpha`` on raw ones. Both are reported so neither reading is ambiguous.

    Args:
        law: The instantiated law.
        params: The fitted parameter vector.
        reference: Normalization constant per predictor.

    Returns:
        A mapping from parameter name to its raw-scale value.

    Examples:
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> raw = raw_scale_params(law, np.array([1.0, 2.0, 3.0, 0.5, 0.25]), np.array([100.0, 16.0]))
        >>> round(raw["A"], 4), round(raw["alpha"], 4)
        (20.0, 0.5)
    """
    values = dict(zip(law.param_names, (float(v) for v in params), strict=True))
    exponents = params[law.n_linear :]
    scales = {"multiplicative-power": float(np.prod(reference**exponents))}
    for i, name in enumerate(law.param_names[1 : law.n_linear]):
        if law.law == "separable-power":
            values[name] = values[name] * float(reference[i] ** exponents[i])
        else:
            values[name] = values[name] * scales[law.law]
    return values


class ScalingLawModel:
    """A fitted scaling law for every recognized target, with uncertainty.

    Instances are produced by :func:`simple_scaling_laws.fit` and by :meth:`load`; they are not normally
    constructed directly.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        schema: Schema,
        law: LawInstance,
        fits: dict[str, TargetFit],
        draws: dict[str, Uncertainty],
        diagnostics: dict[str, Any],
    ):
        """Assemble a fitted model from its parts.

        Args:
            manifest: Provenance and design summary.
            schema: The resolved column roles.
            law: The instantiated law.
            fits: Point estimates, keyed by target.
            draws: Uncertainty draws, keyed by target.
            diagnostics: Correlations, comparisons and warnings.
        """
        self._manifest = manifest
        self._schema = schema
        self._law = law
        self._fits = fits
        self._draws = draws
        self._diagnostics = diagnostics
        # The law is fit only over predictors that actually varied; every predictor the input
        # declared is still tracked, so a prediction at a different value of a held-fixed predictor
        # is correctly reported as extrapolation rather than silently ignored.
        self._domain_predictors = tuple(manifest["predictors"])
        self._reference = np.array(
            [manifest["predictors"][name]["reference"] for name in law.predictors], dtype=float
        )

    def __repr__(self) -> str:
        """One-line summary of the fitted model."""
        return (
            f"ScalingLawModel(law={self._law.law!r}, targets={len(self._fits)}, "
            f"predictors={list(self._law.predictors)}, "
            f"n_configurations={self._manifest['n_configurations']}, "
            f"n_training_runs={self._manifest['n_training_runs']})"
        )

    @property
    def manifest(self) -> dict[str, Any]:
        """Provenance and design summary."""
        return self._manifest

    @property
    def schema(self) -> Schema:
        """The resolved column roles."""
        return self._schema

    @property
    def law(self) -> LawInstance:
        """The instantiated scaling law."""
        return self._law

    @property
    def targets(self) -> tuple[str, ...]:
        """Every fitted target, primary target first."""
        primary = self._schema.primary_target
        return (primary, *(t for t in self._fits if t != primary))

    @property
    def primary_target(self) -> str:
        """The target used as the reference for cross-metric diagnostics."""
        return self._schema.primary_target

    @property
    def predictors(self) -> tuple[str, ...]:
        """Every predictor column the input declared, model-size first.

        Predictions must supply all of them. A predictor that took a single value across the whole experiment
        is not part of the fitted law -- there was nothing to estimate -- but it is still tracked here so that
        predicting at a different value is flagged as extrapolation.
        """
        return self._domain_predictors

    @property
    def fitted_predictors(self) -> tuple[str, ...]:
        """The predictor columns that actually entered the fitted law."""
        return self._law.predictors

    @property
    def fits(self) -> dict[str, TargetFit]:
        """Point estimates and fit quality, keyed by target."""
        return dict(self._fits)

    @property
    def draws(self) -> dict[str, Uncertainty]:
        """Uncertainty draws, keyed by target."""
        return dict(self._draws)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Correlations, cross-target comparisons, fit quality and warnings."""
        return self._diagnostics

    @property
    def warnings(self) -> tuple[Note, ...]:
        """Every warning raised while fitting."""
        return tuple(Note.from_dict(w) for w in self._diagnostics.get("warnings", []))

    @property
    def observed_domain(self) -> dict[str, dict[str, float]]:
        """The observed minimum, maximum and normalization reference of each predictor."""
        return {name: dict(values) for name, values in self._manifest["predictors"].items()}

    def params(self, target: str) -> dict[str, float]:
        """Point-estimate parameters for one target, in human reading order.

        These are in the *normalized* parameterization the fit uses, where each predictor is divided
        by the reference scale in :attr:`observed_domain`. Exponents are unaffected by that choice
        and can be read directly; amplitudes are on the normalized scale, and the artifact's
        ``fits.json`` additionally records ``params_raw_scale`` for the raw-unit reading.

        For a target confined to ``[0, 1]`` the parameters are also on the **logit** scale it was fit
        on, so its offset is an asymptote in logit units -- pass it through
        :func:`simple_scaling_laws.metrics.from_fitting_scale`, or read the asymptote off
        :meth:`predict` at a large scale, to get it back in the metric's own units.

        Args:
            target: The target column name.

        Returns:
            A mapping from parameter name to value.
        """
        fit = self._fit(target)
        values = dict(zip(self._law.param_names, (float(v) for v in fit.params), strict=True))
        return {name: values[name] for name in self._law.display_names}

    def conf_int(self, target: str, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Bootstrap confidence intervals for one target's parameters.

        Args:
            target: The target column name.
            level: Central interval mass, e.g. ``0.95``.

        Returns:
            A mapping from parameter name to a ``(lower, upper)`` tuple.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.
        """
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lower, upper = self._draws[target].quantiles([tail, 1.0 - tail])
        by_name = {name: (float(lower[i]), float(upper[i])) for i, name in enumerate(self._law.param_names)}
        return {name: by_name[name] for name in self._law.display_names}

    def exponents(self, target: str) -> dict[str, float]:
        """The scaling exponents of one target's fitted law."""
        return {
            name: float(value)
            for name, kind, value in zip(
                self._law.param_names, self._law.param_kinds, self._fit(target).params, strict=True
            )
            if kind == EXPONENT
        }

    def metric_correlations(self) -> pl.DataFrame:
        """Run-level associations between the primary target and every other target.

        Returns:
            One row per non-primary target, with run-level Pearson and Spearman correlations, a
            bootstrap interval, and -- for comparison only -- the inflated correlation obtained by
            treating every evaluation row as an independent observation.
        """
        rows = []
        for target, values in self._diagnostics.get("metric_correlations", {}).items():
            interval = values.get("pearson_ci") or [None, None]
            rows.append(
                {
                    "target": target,
                    "n_runs": values["n_runs"],
                    "pearson": values["pearson"],
                    "pearson_q025": interval[0],
                    "pearson_q975": interval[1],
                    "spearman": values["spearman"],
                    "evaluation_row_pearson": values["evaluation_row_pearson"],
                    "n_evaluation_rows": values["n_evaluation_rows"],
                }
            )
        schema = {
            "target": pl.String,
            "n_runs": pl.Int64,
            "pearson": pl.Float64,
            "pearson_q025": pl.Float64,
            "pearson_q975": pl.Float64,
            "spearman": pl.Float64,
            "evaluation_row_pearson": pl.Float64,
            "n_evaluation_rows": pl.Int64,
        }
        return pl.DataFrame(rows, schema=schema)

    def summary(self) -> str:
        """A short human-readable description of the fit, for the CLI and for logs."""
        lines = [
            f"law: {self._law.law}  ({', '.join(self._law.predictors)})",
            f"design: {self._manifest['n_configurations']} configuration(s), "
            f"{self._manifest['n_training_runs']} training run(s), "
            f"{self._manifest['n_evaluation_rows']} evaluation row(s)",
            "targets:",
        ]
        for target in self.targets:
            fit = self._fits[target]
            values = ", ".join(f"{k}={v:.4g}" for k, v in self.params(target).items())
            marker = " (primary)" if target == self.primary_target else ""
            scale = scale_description(self.target_kind(target))
            suffix = "" if scale == "identity" else f"  [fit on the {scale} scale]"
            lines.append(f"  {target}{marker}: {values}{suffix}")
            lines.append(
                f"      run_sd={fit.run_sd:.4g}  "
                f"R2={_format(fit.goodness_of_fit.get('r_squared'))}  "
                f"rmse={_format(fit.goodness_of_fit.get('rmse'))}"
            )
        notes = self.warnings
        lines.append(f"warnings: {len(notes)}")
        lines.extend(f"  {note}" for note in notes)
        return "\n".join(lines)

    def domain_position(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where requested points sit relative to the observed predictor domain.

        Args:
            points: Raw predictor values, shape ``(n_points, n_predictors)``.

        Returns:
            A ``(is_extrapolation, distance)`` pair. ``distance`` is the largest excursion beyond
            the observed range of any predictor, measured in units of that predictor's observed log
            range, and is zero for interior points. A distance of ``1.0`` means the point sits one
            full observed range beyond the edge.
        """
        domain = self.observed_domain
        distances = np.zeros(points.shape[0], dtype=float)
        for j, name in enumerate(self._domain_predictors):
            low, high = np.log(domain[name]["min"]), np.log(domain[name]["max"])
            span = high - low
            value = np.log(points[:, j])
            if span <= 0:
                excursion = np.where(np.isclose(value, low), 0.0, np.inf)
            else:
                excursion = np.maximum(0.0, np.maximum(value - high, low - value)) / span
            distances = np.maximum(distances, excursion)
        return distances > 0, distances

    def predict(
        self,
        points: Any,
        kind: str = "mean",
        targets: Sequence[str] | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Predict target values at new scales, with uncertainty.

        Args:
            points: A dataframe, parquet/CSV path, or mapping of predictor column to values. An
                optional ``point_id`` column is carried through; otherwise identifiers are the row
                positions.
            kind: ``"mean"`` for the distribution of the expected scaling curve ``f(N, D)`` -- what
                this training procedure achieves on average at that scale -- or ``"new-run"`` to add
                training-run stochasticity, i.e. what a single newly trained model might achieve.
            targets: Which targets to predict. Defaults to all of them.
            quantiles: Quantiles to report.
            seed: Seed for the ``new-run`` noise draws. Defaults to the seed recorded at fit time,
                so repeated calls agree.

        Returns:
            One row per requested point: ``point_id``, the predictor values, a ``domain`` label,
            an ``extrapolation_distance``, and one column per target and quantile.

        Raises:
            PredictionError: If ``kind`` is unknown, a requested target was not fit, or the points
                are missing a predictor column.

        Examples:
            >>> from simple_scaling_laws import fit
            >>> from simple_scaling_laws.simulate import simulate_runs
            >>> frame = simulate_runs(
            ...     {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}},
            ...     run_sd=0.01,
            ...     eval_sd=0.02,
            ...     seed=0,
            ... )
            >>> model = fit(frame, n_draws=200, seed=0)
            >>> points = pl.DataFrame(
            ...     {
            ...         "point_id": ["interior", "bigger"],
            ...         "model_size__n_params": [1e7, 1e10],
            ...         "dataset_size__n_subjects": [1e4, 1e5],
            ...     }
            ... )
            >>> predictions = model.predict(points)
            >>> predictions["domain"].to_list()
            ['interpolation', 'extrapolation']
            >>> sorted(c for c in predictions.columns if c.startswith("test_loss"))
            ['test_loss__ce__median', 'test_loss__ce__q025', 'test_loss__ce__q975']

            A ``new-run`` prediction is wider than a ``mean`` prediction, because it also carries
            the training stochasticity of a single new model:

            >>> new_run = model.predict(points, kind="new-run")
            >>> def width(frame):
            ...     return frame["test_loss__ce__q975"] - frame["test_loss__ce__q025"]
            >>> bool((width(new_run) > width(predictions)).all())
            True
        """
        if kind not in PREDICTION_KINDS:
            raise PredictionError(f"kind must be one of {list(PREDICTION_KINDS)}, got {kind!r}")
        requested = list(targets) if targets is not None else list(self.targets)
        unknown = [t for t in requested if t not in self._fits]
        if unknown:
            raise PredictionError(f"No fit for target(s) {unknown}; available: {list(self.targets)}")
        quantiles = [float(q) for q in quantiles]
        if any(not 0 <= q <= 1 for q in quantiles):
            raise PredictionError(f"quantiles must lie in [0, 1], got {quantiles}")

        frame = self.prepare_points(points)
        raw = frame.select(self._domain_predictors).to_numpy().astype(float)
        log_x = self.fitting_predictors(frame)

        is_extrapolation, distance = self.domain_position(raw)
        if bool(is_extrapolation.any()):
            warnings.warn(
                f"{int(is_extrapolation.sum())} of {raw.shape[0]} prediction point(s) lie outside "
                "the observed scaling domain. Their intervals reflect parameter uncertainty only "
                "and cannot account for the scaling law being the wrong functional form out there.",
                UserWarning,
                stacklevel=2,
            )

        rng = np.random.default_rng(self._manifest.get("seed", 0) if seed is None else seed)
        columns: dict[str, Any] = {
            POINT_ID: frame[POINT_ID],
            **{name: frame[name] for name in self._domain_predictors},
            "domain": np.where(is_extrapolation, "extrapolation", "interpolation"),
            "extrapolation_distance": distance,
        }
        for target in requested:
            values = self.target_draws(target, log_x, kind=kind, rng=rng)
            for q in quantiles:
                columns[f"{target}__{quantile_suffix(q)}"] = np.quantile(values, q, axis=0)
        return pl.DataFrame(columns)

    def prepare_points(self, points: Any) -> pl.DataFrame:
        """Validate a prediction request into a frame with a ``point_id`` and every predictor.

        Args:
            points: A dataframe, parquet/CSV path, or mapping of predictor column to values.

        Returns:
            The validated frame.

        Raises:
            PredictionError: If a predictor column is missing or a value is not positive and finite.
        """
        frame = _points_frame(points, self._domain_predictors)
        raw = frame.select(self._domain_predictors).to_numpy().astype(float)
        if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
            raise PredictionError("Predictor values must be finite and strictly positive")
        return frame

    def fitting_predictors(self, frame: pl.DataFrame) -> np.ndarray:
        """Log-normalized values of this model's *fitted* predictors, ready for the law.

        Args:
            frame: A frame from :meth:`prepare_points`.

        Returns:
            Array of shape ``(n_points, n_fitted_predictors)``.
        """
        fitted = frame.select(self._law.predictors).to_numpy().astype(float)
        return np.log(fitted) - np.log(self._reference)

    def target_draws(
        self,
        target: str,
        log_x: np.ndarray,
        kind: str = "mean",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Predicted values for one target, one row per bootstrap draw, on the target's own scale.

        This is the primitive behind :meth:`predict`, exposed because comparing systems needs the
        draws themselves rather than their quantiles.

        Args:
            target: The target column name.
            log_x: Log-normalized predictors for the fitted law, shape ``(n_points, n_fitted)``.
            kind: ``"mean"`` or ``"new-run"``.
            rng: Random source for the ``new-run`` noise.

        Returns:
            Array of shape ``(n_draws, n_points)``.
        """
        draws = self._draws[target]
        values = self._law.evaluate_many(draws.params, log_x)
        if kind == "new-run":
            values = values + _run_noise(rng or np.random.default_rng(0), draws, values.shape)
        # Training-run noise is added on the fitting scale, where the model is additive, and only
        # then mapped back -- so a bounded metric's interval cannot leave its bounds.
        return from_fitting_scale(values, self.target_kind(target))

    def target_kind(self, target: str) -> MetricKind:
        """How one target behaves: which direction is better, and what scale it is fit on."""
        return self._schema.target(target).kind

    def save(self, path: str | Path) -> Path:
        """Write this model to a ``.slaw`` artifact directory.

        Args:
            path: Destination directory.

        Returns:
            The resolved artifact path.
        """
        fits = {target: fit.to_dict(self._law, self._reference) for target, fit in self._fits.items()}
        draws: dict[str, np.ndarray] = {}
        for target, uncertainty in self._draws.items():
            for name, values in to_arrays(self._law, uncertainty).items():
                draws[f"{target}/{name}"] = values
        return artifact_io.write_artifact(path, self._manifest, fits, draws, self._diagnostics)

    @classmethod
    def load(cls, path: str | Path) -> ScalingLawModel:
        """Read a model back from a ``.slaw`` artifact directory.

        Args:
            path: The artifact directory.

        Returns:
            The reconstructed model, which predicts identically to the one that was saved.
        """
        manifest, fits_data, draws_data, diagnostics = artifact_io.read_artifact(path)
        law = LawInstance.from_dict(manifest["law"])
        schema = Schema.from_dict(manifest["schema"])
        fits = {name: TargetFit.from_dict(name, data, law) for name, data in fits_data.items()}
        draws = {}
        for name, fit in fits.items():
            arrays = {
                key.split("/", 1)[1]: value for key, value in draws_data.items() if key.startswith(f"{name}/")
            }
            draws[name] = from_arrays(law, arrays, fit.uncertainty_method, fit.n_failed_draws)
        return cls(manifest, schema, law, fits, draws, diagnostics)

    def _fit(self, target: str) -> TargetFit:
        """Look up one target's fit, with a helpful error when it is absent."""
        if target not in self._fits:
            raise KeyError(f"No fit for target {target!r}; available: {list(self.targets)}")
        return self._fits[target]


def _format(value: float | None) -> str:
    """Format an optional float for the text summary."""
    return "n/a" if value is None else f"{value:.4g}"


def _run_noise(rng: np.random.Generator, draws: Uncertainty, shape: tuple[int, int]) -> np.ndarray:
    """Training-run noise for a ``new-run`` prediction.

    The *shape* of the distribution comes from the data whenever enough run-level deviations were actually
    observed -- they are standardized and resampled, so no normality is assumed -- while the *scale* always
    comes from the bootstrap draws, so uncertainty about how variable training runs are is carried rather than
    fixed at its point estimate. With too few observed deviations to resample there is no empirical shape to
    borrow, and the noise falls back to a normal draw at the same bootstrapped scale.
    """
    deviations = draws.run_deviations
    if deviations.size >= MIN_DEVIATIONS_FOR_RESAMPLING:
        spread = float(np.std(deviations, ddof=1))
        if spread > 0:
            standardized = deviations / spread
            return rng.choice(standardized, size=shape, replace=True) * draws.run_sd[:, None]
    return rng.normal(0.0, 1.0, size=shape) * draws.run_sd[:, None]


def _points_frame(points: Any, predictors: Sequence[str]) -> pl.DataFrame:
    """Coerce a prediction request into a frame with the predictors and a ``point_id``."""
    if isinstance(points, dict):
        frame = pl.DataFrame(points)
    else:
        try:
            frame = read_table(points)
        except DataError as exc:
            raise PredictionError(str(exc)) from exc
    missing = [name for name in predictors if name not in frame.columns]
    if missing:
        raise PredictionError(
            f"Prediction points are missing predictor column(s) {missing}. Expected {list(predictors)}."
        )
    if POINT_ID not in frame.columns:
        frame = frame.with_row_index(POINT_ID).with_columns(pl.col(POINT_ID).cast(pl.String))
    return frame.with_columns(pl.col(name).cast(pl.Float64) for name in predictors)
