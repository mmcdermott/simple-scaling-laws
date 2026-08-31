"""The top-level fitting entry point.

One call fits every recognized target, estimates its uncertainty, computes the cross-metric
diagnostics, and returns a self-contained :class:`~simple_scaling_laws.model.ScalingLawModel`.
Nothing about the statistics is a required argument: the law family is the only choice a user
normally makes.
"""

from __future__ import annotations

import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from . import artifact as artifact_io
from . import diagnostics as diagnostics_mod
from .data import Dataset, build_dataset
from .fitting import (
    MIN_REPLICATE_DOF,
    VarianceComponents,
    compute_weights,
    constant_fit,
    fit_curve,
    is_constant,
    leverage,
    parameter_bounds,
    replicate_run_variance,
    residual_run_variance,
)
from .laws import DEFAULT_LAW, LawInstance, build_law
from .model import ScalingLawModel, TargetFit
from .notes import Note
from .uncertainty import DEFAULT_DRAWS, Uncertainty, bootstrap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

    from .data import TargetObservations

#: Keys accepted in a configuration file.
CONFIG_KEYS: frozenset[str] = frozenset({"law", "columns", "primary_target", "targets", "n_draws", "seed"})


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    r"""Read an optional YAML configuration file.

    A configuration only ever needs to name the law; column roles are inferred unless the input
    table uses non-conventional names.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed settings.

    Raises:
        ConfigError: If the file is not a mapping or contains unknown keys.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "config.yaml"
        >>> _ = path.write_text(
        ...     "law: separable-power\ncolumns:\n  training_run_id: run\n"
        ... )
        >>> load_config(path)
        {'law': 'separable-power', 'columns': {'training_run_id': 'run'}}
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping, got {type(data).__name__}")
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise ConfigError(f"Unknown configuration key(s) {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}")
    return data


def target_seed(seed: int, target: str) -> int:
    """The random seed used for one target's bootstrap.

    Derived from the target's *name* rather than its position, so that the same target draws the
    same cluster multipliers in two different experiments even when those experiments report
    different sets of metrics. That is what lets :func:`simple_scaling_laws.compare` difference two
    fits' draws and get the variance of the difference right rather than the sum of two variances.

    Args:
        seed: The experiment-level seed.
        target: The target column name.

    Returns:
        A seed in ``[0, 2**32)``.

    Examples:
        >>> target_seed(0, "test_loss__ce") == target_seed(0, "test_loss__ce")
        True
        >>> target_seed(0, "test_loss__ce") == target_seed(0, "test_metric__auroc")
        False
    """
    return (seed + zlib.crc32(target.encode())) % (2**32)


def _variance_components(
    observations: TargetObservations,
    residuals: np.ndarray,
    leverages: np.ndarray,
    per_run_var: np.ndarray,
    n_params: int,
) -> VarianceComponents:
    """Estimate both variance components and choose which run-variance estimate to use.

    The replicate-based estimate is preferred whenever independently trained models exist at the
    same scale, because it measures training stochasticity directly. Without replicates the only
    available estimate is the scatter around the fitted curve, which also contains the law's own
    misspecification -- an upward bias, and therefore the safe direction to err in.
    """
    replicate, replicate_dof = replicate_run_variance(observations, observations.eval_var_independent)
    residual, residual_dof = residual_run_variance(residuals, leverages, per_run_var, n_params)
    if replicate is not None and replicate_dof >= MIN_REPLICATE_DOF:
        run_var, source = replicate, "replicates"
    elif residual is not None:
        run_var, source = residual, "residual"
    elif replicate is not None:
        run_var, source = replicate, "replicates"
    else:
        run_var, source = 0.0, "none"
    return VarianceComponents(
        eval_var=observations.eval_var,
        eval_var_independent=observations.eval_var_independent,
        run_var=float(run_var),
        run_var_replicate=replicate,
        run_var_residual=residual,
        replicate_dof=replicate_dof,
        residual_dof=residual_dof,
        source=source,
    )


def _fit_target(
    dataset: Dataset,
    law: LawInstance,
    target: str,
    n_draws: int,
    seed: int,
) -> tuple[TargetFit, Uncertainty, list[Note]]:
    """Fit one target: two weighted passes, variance components, then the bootstrap."""
    observations = dataset.observations[target]
    signed = dataset.schema.target(target).signed_amplitude
    log_x = dataset.log_x[:, dataset.column_indices(law.predictors)][observations.config_index]
    y = observations.mean
    bounds = parameter_bounds(law, y, signed_amplitude=signed)
    per_run_var = observations.eval_var_independent / np.maximum(observations.n_eval, 1)

    if is_constant(y) or dataset.n_configurations < 2:
        return _constant_target_fit(dataset, law, target, observations, y, per_run_var, n_draws, seed)

    unweighted = np.ones_like(y)
    first = fit_curve(law, log_x, y, unweighted, bounds)
    components = _variance_components(
        observations,
        y - law.evaluate(first.params, log_x),
        leverage(law, first.params, log_x, unweighted),
        per_run_var,
        law.n_params,
    )

    weights = compute_weights(components.run_var, per_run_var)
    fit = fit_curve(
        law, log_x, y, weights, bounds, starts=[first.params, *_seed_from(law, log_x, y, weights, bounds)]
    )
    residuals = y - law.evaluate(fit.params, log_x)
    components = _variance_components(
        observations, residuals, leverage(law, fit.params, log_x, weights), per_run_var, law.n_params
    )

    uncertainty = bootstrap(
        law,
        log_x,
        y,
        weights,
        bounds,
        fit.params,
        observations.config_index,
        per_run_var,
        components.run_sd,
        n_draws=n_draws,
        seed=seed,
        target=target,
    )
    goodness = diagnostics_mod.goodness_of_fit(y, law.evaluate(fit.params, log_x), weights, law.n_params)
    target_fit = TargetFit(
        target=target,
        role=dataset.schema.target(target).role,
        signed_amplitude=signed,
        params=fit.params,
        variance_components=components.to_dict(),
        optimizer={
            "success": fit.success,
            "cost": fit.cost,
            "message": fit.message,
            "n_function_evaluations": fit.n_function_evaluations,
            "n_starts": fit.n_starts,
        },
        goodness_of_fit=goodness,
        uncertainty_method=uncertainty.method,
        n_failed_draws=uncertainty.n_failed,
        pairing={
            "method": uncertainty.method,
            "n_clusters": observations.n_configurations,
            "n_draws": n_draws,
            "seed": seed,
        },
    )
    notes = list(uncertainty.notes)
    notes.extend(diagnostics_mod.fit_notes(target, law, fit, bounds, uncertainty, components))
    return target_fit, uncertainty, notes


def _constant_target_fit(
    dataset: Dataset,
    law: LawInstance,
    target: str,
    observations: TargetObservations,
    y: np.ndarray,
    per_run_var: np.ndarray,
    n_draws: int,
    seed: int,
) -> tuple[TargetFit, Uncertainty, list[Note]]:
    """Handle a target with no estimable curve: an exact constant fit, no optimizer, no refits.

    Two very different situations land here. A target that is genuinely flat -- a saturated metric,
    or a mistake in the input -- has nothing at all to estimate. An experiment run at a *single*
    scaling configuration has no curve either, but it does still say something useful: how much the
    level varies from one trained model to the next. That part is estimated properly, so
    ``predict`` at the observed scale still returns an honest interval and ``new-run`` still carries
    training stochasticity. Only the scaling behavior is unavailable, and that is what the warning
    says.

    The offset's uncertainty comes from an ordinary nonparametric bootstrap over training runs,
    which is exactly right here: with one configuration there is no design to preserve, so runs are
    the sampling unit and resampling them directly needs no further assumptions.
    """
    rng = np.random.default_rng(seed)
    fit_result = constant_fit(law, y)
    resampled = y[rng.integers(0, y.size, size=(n_draws, y.size))]

    eval_noise = float(np.mean(per_run_var))
    resampled_var = resampled.var(axis=1, ddof=1) if y.size > 1 else np.zeros(n_draws)
    run_sd_draws = np.sqrt(np.maximum(0.0, resampled_var - eval_noise))
    run_var = max(0.0, float(np.var(y, ddof=1)) - eval_noise) if y.size > 1 else 0.0
    deviations = y - float(np.mean(y)) if dataset.n_configurations < 2 and y.size > 1 else np.zeros(0)

    components = VarianceComponents(
        eval_var=observations.eval_var,
        eval_var_independent=observations.eval_var_independent,
        run_var=run_var,
        run_var_replicate=run_var if y.size > 1 else None,
        run_var_residual=None,
        replicate_dof=max(0, y.size - 1) if dataset.n_configurations < 2 else 0,
        residual_dof=0,
        source="replicates" if run_var > 0 else "none",
    )
    params = np.tile(fit_result.params, (n_draws, 1))
    params[:, 0] = resampled.mean(axis=1)
    uncertainty = Uncertainty(
        params=params,
        run_sd=run_sd_draws,
        run_deviations=deviations,
        method="constant",
        n_failed=0,
    )
    reason = (
        "the experiment contains a single scaling configuration"
        if dataset.n_configurations < 2
        else "it takes the same value at every scale"
    )
    note = Note(
        "constant_target",
        "warning",
        f"No scaling law could be estimated for {target!r} because {reason}. It is reported as a "
        "constant at the observed level; its exponents are zero and carry no meaning, and any "
        "prediction at a different scale is unsupported.",
        {
            "target": target,
            "value": float(np.mean(y)),
            "configurations": [
                dict(zip(dataset.schema.predictors, (float(v) for v in row), strict=True))
                for row in dataset.config_values
            ],
            "n_configurations": dataset.n_configurations,
            "n_runs": int(y.size),
        },
    )
    target_fit = TargetFit(
        target=target,
        role=dataset.schema.target(target).role,
        signed_amplitude=dataset.schema.target(target).signed_amplitude,
        params=fit_result.params,
        variance_components=components.to_dict(),
        optimizer={
            "success": True,
            "cost": 0.0,
            "message": fit_result.message,
            "n_function_evaluations": 0,
            "n_starts": 0,
        },
        goodness_of_fit=diagnostics_mod.goodness_of_fit(
            y, np.full_like(y, fit_result.params[0]), np.ones_like(y), 1
        ),
        uncertainty_method="constant",
        n_failed_draws=0,
        pairing={"method": "constant", "n_clusters": 0, "n_draws": n_draws, "seed": seed},
    )
    return target_fit, uncertainty, [note]


def _seed_from(law, log_x, y, weights, bounds):
    """Grid seeds for the reweighted pass, so reweighting cannot strand the fit in a worse basin."""
    from .fitting import GridSeeder

    return [params for _, params in GridSeeder(law, log_x, weights, bounds).seeds(y, n_seeds=2)]


def _manifest(
    dataset: Dataset,
    law: LawInstance,
    source: Any,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Assemble the artifact manifest."""
    domain = dataset.observed_domain()
    primary = dataset.observations[dataset.schema.primary_target]
    return {
        "format_version": artifact_io.FORMAT_VERSION,
        "package_version": artifact_io.package_version(),
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "law": law.to_dict(),
        "schema": dataset.schema.to_dict(),
        "predictors": {
            name: {
                "role": dataset.schema.predictor_role(name),
                "in_law": name in law.predictors,
                **domain[name],
            }
            for name in dataset.schema.predictors
        },
        "targets": {
            t.name: {
                "role": t.role,
                "signed_amplitude": t.signed_amplitude,
                "kind": t.kind.to_dict(),
            }
            for t in dataset.schema.targets
            if t.name in dataset.observations
        },
        "primary_target": dataset.schema.primary_target,
        "configurations": [
            dict(zip(dataset.schema.predictors, (float(v) for v in row), strict=True))
            for row in dataset.config_values
        ],
        "n_configurations": dataset.n_configurations,
        "n_training_runs": dataset.n_runs,
        "n_evaluation_rows": dataset.n_rows,
        "n_test_sets": dataset.n_test_sets,
        "runs_per_configuration": {
            "min": int(dataset.runs_per_config.min()),
            "max": int(dataset.runs_per_config.max()),
        },
        "evaluations_per_run": {
            "min": int(primary.n_eval.min()),
            "max": int(primary.n_eval.max()),
        },
        "source": {"path": str(source) if isinstance(source, str | Path) else None},
        "seed": seed,
        "n_draws": n_draws,
    }


def fit(
    source: Any,
    law: str = DEFAULT_LAW,
    columns: Mapping[str, str | Iterable[str]] | None = None,
    primary_target: str | None = None,
    targets: Mapping[str, Any] | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> ScalingLawModel:
    """Fit a scaling law to every recognized target in an experiment table.

    Args:
        source: A parquet/CSV path or an in-memory dataframe of evaluation rows.
        law: The scaling-law family, see :func:`simple_scaling_laws.laws.available_laws`.
        columns: Optional column-role overrides, e.g. ``{"training_run_id": "run"}``.
        primary_target: Optional explicit primary target; defaults to the single test loss.
        targets: Optional explicit per-target descriptions, e.g.
            ``{"test_metric__custom": {"support": "unit", "direction": "higher"}}``. A target
            confined to ``[0, 1]`` is fit on its logit so its asymptote cannot leave that range;
            well-known metric names are recognized without being named here.
        config: Optional YAML path or mapping supplying any of the above. It fills in only the
            arguments left at their default value, so anything passed explicitly wins.
        n_draws: Number of bootstrap draws to retain per target.
        seed: Seed for every random draw, so a fit is exactly reproducible.

    Returns:
        The fitted :class:`~simple_scaling_laws.model.ScalingLawModel`.

    Examples:
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> frame = simulate_runs(
        ...     {
        ...         "test_loss__cross_entropy": {
        ...             "E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25
        ...         },
        ...         "test_metric__auroc": {
        ...             "E": 0.9, "A": -0.2, "alpha": 0.4, "B": -0.1, "beta": 0.3
        ...         },
        ...     },
        ...     run_sd=0.01,
        ...     eval_sd=0.02,
        ...     seed=0,
        ... )
        >>> model = fit(frame, n_draws=200, seed=0)
        >>> model.targets
        ('test_loss__cross_entropy', 'test_metric__auroc')

        Every target gets its own parameters and its own draws:

        >>> params = model.params("test_loss__cross_entropy")
        >>> round(params["E"], 1), round(params["alpha"], 2)
        (1.0, 0.3)
        >>> model.draws["test_metric__auroc"].params.shape
        (200, 5)

        A metric that improves with scale gets a negative amplitude, so it approaches its asymptote
        from below:

        >>> bool(model.params("test_metric__auroc")["A"] < 0)
        True
    """
    settings = dict(load_config(config) if isinstance(config, str | Path) else (config or {}))
    law = settings.get("law", law) if law == DEFAULT_LAW else law
    columns = columns if columns is not None else settings.get("columns")
    primary_target = primary_target if primary_target is not None else settings.get("primary_target")
    targets = targets if targets is not None else settings.get("targets")
    n_draws = int(settings.get("n_draws", n_draws)) if n_draws == DEFAULT_DRAWS else int(n_draws)
    seed = int(settings.get("seed", seed)) if seed == 0 else int(seed)

    dataset = build_dataset(source, columns=columns, primary_target=primary_target, targets=targets)
    notes: list[Note] = list(dataset.notes)

    varying = dataset.varying_predictors()
    dropped = [name for name in dataset.schema.predictors if name not in varying]
    if varying:
        instance = build_law(
            law,
            [n for n in dataset.schema.model_size if n in varying],
            [n for n in dataset.schema.dataset_size if n in varying],
        )
    else:
        instance = build_law(law, dataset.schema.model_size, dataset.schema.dataset_size)
    if dropped:
        notes.append(
            Note(
                "constant_predictor_dropped",
                "warning",
                f"Predictor(s) {dropped} take a single value across the whole experiment, so their "
                "terms are indistinguishable from the law's offset and were left out of the fit. "
                "Predictions at any other value of those predictors are pure extrapolation.",
                {"predictors": dropped, "fitted_predictors": list(instance.predictors)},
            )
        )
    notes.extend(diagnostics_mod.design_notes(dataset, instance))

    fits: dict[str, TargetFit] = {}
    draws: dict[str, Uncertainty] = {}
    for target in dataset.observations:
        target_fit, uncertainty, target_notes = _fit_target(
            dataset, instance, target, n_draws=n_draws, seed=target_seed(seed, target)
        )
        fits[target] = target_fit
        draws[target] = uncertainty
        notes.extend(target_notes)

    params = {target: fit.params for target, fit in fits.items()}
    diagnostics = {
        "metric_correlations": diagnostics_mod.metric_correlations(dataset, seed=seed),
        "scaling_similarity": diagnostics_mod.scaling_similarity(dataset, instance, params, list(fits)),
        "fit_quality": {target: fit.goodness_of_fit for target, fit in fits.items()},
        "uncertainty": {
            target: {
                "method": uncertainty.method,
                "n_draws": uncertainty.n_draws,
                "n_failed_draws": uncertainty.n_failed,
                "n_run_deviations": int(uncertainty.run_deviations.size),
            }
            for target, uncertainty in draws.items()
        },
        "paired_evaluation": {
            target: {
                "eval_sd": float(np.sqrt(observations.eval_var)),
                "eval_sd_independent": float(np.sqrt(observations.eval_var_independent)),
                "eval_sd_shared": float(
                    np.sqrt(max(0.0, observations.eval_var - observations.eval_var_independent))
                ),
                "eval_pair_correlation": observations.eval_pair_correlation,
                "n_shared_pairs": observations.n_shared_pairs,
            }
            for target, observations in dataset.observations.items()
        },
        "warnings": [note.to_dict() for note in notes],
    }
    manifest = _manifest(dataset, instance, source, n_draws=n_draws, seed=seed)
    return ScalingLawModel(manifest, dataset.schema, instance, fits, draws, diagnostics)
