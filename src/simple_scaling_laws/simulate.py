"""Synthetic data generation from a known scaling law.

This exists so the package can be tested against ground truth, and so the documentation can show a
complete worked example without shipping a data file. It generates exactly the input shape the
fitter expects: long over evaluation resamples, wide over metrics, with explicit run and test-set
identifiers.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .laws import build_law

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Default predictor column names.
DEFAULT_PREDICTORS: tuple[str, str] = ("model_size__n_params", "dataset_size__n_subjects")


def _per_target(value: float | Mapping[str, float], targets: Sequence[str]) -> dict[str, float]:
    """Broadcast a scalar to every target, or validate a per-target mapping."""
    if isinstance(value, Mapping):
        missing = set(targets) - set(value)
        if missing:
            raise ValueError(f"No value supplied for target(s) {sorted(missing)}")
        return {t: float(value[t]) for t in targets}
    return dict.fromkeys(targets, float(value))


def simulate_runs(
    params: Mapping[str, Mapping[str, float]],
    model_sizes: Sequence[float] = (1e6, 1e7, 1e8),
    dataset_sizes: Sequence[float] = (1e3, 1e4, 1e5),
    law: str = "separable-power",
    runs_per_config: int = 2,
    evaluations_per_run: int = 5,
    run_sd: float | Mapping[str, float] = 0.0,
    eval_sd: float | Mapping[str, float] = 0.0,
    shared_eval_fraction: float = 0.0,
    paired_test_sets: bool = True,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
    seed: int = 0,
) -> pl.DataFrame:
    """Generate evaluation records from a known scaling law.

    Args:
        params: Per target, a mapping of law parameter name to its true value. Parameter names are
            those of the chosen law, e.g. ``E``, ``A``, ``alpha``, ``B``, ``beta``.
        model_sizes: Raw model-size values; crossed with ``dataset_sizes`` to form configurations.
        dataset_sizes: Raw dataset-size values.
        law: Name of the generating law.
        runs_per_config: Independently trained models per configuration.
        evaluations_per_run: Evaluation resamples scored for each trained model.
        run_sd: Training-run noise standard deviation, shared or per target.
        eval_sd: Evaluation noise standard deviation, shared or per target.
        shared_eval_fraction: Fraction of the evaluation *variance* attributable to the test set
            itself, i.e. shared by every model scored on that resample. Only has an effect when
            ``paired_test_sets`` is true.
        paired_test_sets: Whether every trained model is scored on the same resamples (so
            ``test_set_id`` values are reused across models) or on its own.
        predictors: The two predictor column names.
        seed: Seed for the random number generator.

    Returns:
        A Polars dataframe of evaluation rows.

    Raises:
        ValueError: If ``params`` omits a parameter of the chosen law.

    Examples:
        >>> frame = simulate_runs(
        ...     {"test_loss__cross_entropy": {"E": 1.0, "A": 3.0, "alpha": 0.3, "B": 2.0, "beta": 0.2}},
        ...     model_sizes=(1e6, 1e7),
        ...     dataset_sizes=(1e3, 1e4),
        ...     runs_per_config=2,
        ...     evaluations_per_run=3,
        ...     run_sd=0.01,
        ...     eval_sd=0.02,
        ...     seed=0,
        ... )
        >>> frame.shape
        (24, 7)
        >>> frame.columns
        ['training_run_id', 'train_set_id', 'test_set_id', 'optimizer_seed',
         'model_size__n_params', 'dataset_size__n_subjects', 'test_loss__cross_entropy']
        >>> frame["training_run_id"].n_unique(), frame["test_set_id"].n_unique()
        (8, 3)

        With paired test sets, the same resample identifiers appear for every trained model:

        >>> frame.group_by("test_set_id").len().sort("test_set_id")["len"].to_list()
        [8, 8, 8]

        Noiseless data reproduces the law exactly:

        >>> exact = simulate_runs(
        ...     {"test_loss__ce": {"E": 1.0, "A": 3.0, "alpha": 0.5, "B": 0.0, "beta": 0.5}},
        ...     model_sizes=(1e6,),
        ...     dataset_sizes=(1e3,),
        ...     runs_per_config=1,
        ...     evaluations_per_run=1,
        ... )
        >>> exact["test_loss__ce"].to_list()
        [4.0]
    """
    targets = list(params)
    run_sds = _per_target(run_sd, targets)
    eval_sds = _per_target(eval_sd, targets)
    model_sizes, dataset_sizes = tuple(model_sizes), tuple(dataset_sizes)
    instance = build_law(law, predictors[:1], predictors[1:])

    vectors: dict[str, np.ndarray] = {}
    for target, values in params.items():
        missing = set(instance.param_names) - set(values)
        if missing:
            raise ValueError(f"Target {target!r} is missing parameter(s) {sorted(missing)} for law {law!r}")
        vectors[target] = np.array([float(values[name]) for name in instance.param_names])

    # Independent streams, so that changing the number of evaluation resamples does not also change
    # the realized training-run offsets. That keeps two designs genuinely comparable.
    shared_rng, run_rng, eval_rng = np.random.default_rng(seed).spawn(3)
    configs = list(itertools.product(model_sizes, dataset_sizes))
    reference = np.exp(np.log(np.array(configs, dtype=float)).mean(axis=0))
    shared = {
        target: shared_rng.normal(
            0.0, eval_sds[target] * np.sqrt(shared_eval_fraction), evaluations_per_run
        )
        for target in targets
    }

    rows: dict[str, list] = {name: [] for name in ("training_run_id", "train_set_id", "test_set_id")}
    rows["optimizer_seed"] = []
    for name in predictors:
        rows[name] = []
    for target in targets:
        rows[target] = []

    for config_i, (model_size, dataset_size) in enumerate(configs):
        log_x = np.log(np.array([[model_size, dataset_size]], dtype=float)) - np.log(reference)
        truth = {t: float(instance.evaluate(vectors[t], log_x)[0]) for t in targets}
        for replicate in range(runs_per_config):
            run_id = f"run_{config_i:03d}_{replicate:02d}"
            offsets = {t: run_rng.normal(0.0, run_sds[t]) for t in targets}
            for evaluation in range(evaluations_per_run):
                test_set_id = (
                    f"boot_{evaluation:03d}" if paired_test_sets else f"{run_id}_boot_{evaluation:03d}"
                )
                rows["training_run_id"].append(run_id)
                rows["train_set_id"].append(f"train_{config_i:03d}")
                rows["test_set_id"].append(test_set_id)
                rows["optimizer_seed"].append(replicate)
                rows[predictors[0]].append(model_size)
                rows[predictors[1]].append(dataset_size)
                for target in targets:
                    idiosyncratic = eval_sds[target] * np.sqrt(1.0 - shared_eval_fraction)
                    noise = eval_rng.normal(0.0, idiosyncratic) if idiosyncratic > 0 else 0.0
                    if paired_test_sets:
                        noise += shared[target][evaluation]
                    rows[target].append(truth[target] + offsets[target] + noise)
    return pl.DataFrame(rows)
