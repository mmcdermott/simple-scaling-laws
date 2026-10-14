"""Cost laws: what a training run of a given model size and dataset size costs.

A scaling law answers "what loss will this configuration reach?". Allocating a budget needs a second
model, of what each configuration *costs*, so that the loss can be minimized along a curve of equal
cost. That model is not something this package can infer from the losses, and it is not the same for
every experiment, so -- exactly as with the scaling laws themselves -- it comes from a small registry of
named forms, each over one model-size predictor ``N`` and one dataset-size predictor ``D``:

* ``power``: ``C = k * N**a * D**b``. With ``k = 6, a = b = 1`` it is the usual estimate of transformer
  training FLOPs, six per parameter per token.
* ``per-example``: ``C = D * (k * N**a + c)``. Cost per example processed grows with the model, plus an
  overhead ``c`` that does not -- attention over a long context, embeddings, data handling -- so small
  models are dearer than a pure power law says.

A cost model is either **declared**, with every parameter given, or **fitted** from runs that recorded
what they actually cost (a ``compute__*`` column, for instance FLOPs logged by the training loop). Any
subset of a form's parameters can be fixed and the rest fitted. Both routes give the same
:class:`CostModel`, which is all the allocation and design code ever sees.

Measuring beats declaring wherever it is possible. ``D`` here must be what a run *processed* -- tokens
or examples seen, all epochs included -- and a run that stops early processes less than its schedule
said. Logged compute captures that; a formula written down in advance does not.

Examples:
    >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
    >>> f"{float(chinchilla(1e9, 2e10)):.3g}"
    '1.2e+20'

    The dataset size that spends a budget exactly, at a given model size:

    >>> f"{float(chinchilla.dataset_size_for(1.2e20, 1e9)):.3g}"
    '2e+10'
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
from scipy.optimize import lsq_linear, minimize_scalar

from .data import read_table
from .notes import Note
from .schema import ID_ROLES, PREDICTOR_ROLES, SEP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

#: Column-name prefix of a recorded per-run cost.
COMPUTE_ROLE = "compute"

#: A fitted cost model whose predictions miss the recorded costs by more than this root-mean-square
#: error on the log scale (about 10%) is reported, since every allocation built on it inherits the error.
POOR_FIT_LOG_ERROR = 0.1

#: Recorded costs that differ by more than this relative amount within one training run are reported:
#: a run has one cost, and disagreement usually means the column is per evaluation rather than per run.
WITHIN_RUN_TOLERANCE = 1e-6

#: Range searched for an exponent that is being fitted.
EXPONENT_RANGE = (0.05, 4.0)


class CostError(ValueError):
    """Raised for unknown cost forms, incomplete declarations, or records that cannot be fit."""


class _PowerCost:
    """``C = k * N**a * D**b``."""

    name: ClassVar[str] = "power"
    description: ClassVar[str] = "k * N**a * D**b (e.g. k=6, a=b=1: training FLOPs of a transformer)"
    param_names: ClassVar[tuple[str, ...]] = ("k", "a", "b")

    @staticmethod
    def evaluate(params: Mapping[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
        return params["k"] * n ** params["a"] * d ** params["b"]

    @staticmethod
    def dataset_size_for(params: Mapping[str, float], budget: np.ndarray, n: np.ndarray) -> np.ndarray:
        return (budget / (params["k"] * n ** params["a"])) ** (1.0 / params["b"])

    @staticmethod
    def fit(n: np.ndarray, d: np.ndarray, c: np.ndarray, fixed: Mapping[str, float]) -> dict[str, float]:
        """Ordinary least squares on ``log C = log k + a log N + b log D``, fixed terms moved left."""
        columns = {"k": np.ones_like(n), "a": np.log(n), "b": np.log(d)}
        target = np.log(c).copy()
        for name, value in fixed.items():
            target -= (np.log(value) if name == "k" else value) * columns[name]
        free = [name for name in _PowerCost.param_names if name not in fixed]
        design = np.column_stack([columns[name] for name in free])
        if np.linalg.matrix_rank(design) < len(free):
            raise CostError(
                f"The recorded runs cannot identify {free}: a fitted exponent needs its predictor to vary"
            )
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        params = dict(fixed)
        for name, value in zip(free, solution, strict=True):
            params[name] = float(np.exp(value)) if name == "k" else float(value)
        return params


class _PerExampleCost:
    """``C = D * (k * N**a + c)``."""

    name: ClassVar[str] = "per-example"
    description: ClassVar[str] = "D * (k * N**a + c) (per-example cost growing with N, plus a fixed overhead)"
    param_names: ClassVar[tuple[str, ...]] = ("k", "a", "c")

    @staticmethod
    def evaluate(params: Mapping[str, float], n: np.ndarray, d: np.ndarray) -> np.ndarray:
        return d * (params["k"] * n ** params["a"] + params["c"])

    @staticmethod
    def dataset_size_for(params: Mapping[str, float], budget: np.ndarray, n: np.ndarray) -> np.ndarray:
        return budget / (params["k"] * n ** params["a"] + params["c"])

    @staticmethod
    def _solve_linear(
        n: np.ndarray, per_example: np.ndarray, a: float, fixed: Mapping[str, float]
    ) -> tuple[dict[str, float], float]:
        """Given ``a``, solve ``k`` and ``c`` by non-negative least squares on relative error."""
        weight = 1.0 / per_example
        columns = {"k": n**a * weight, "c": weight}
        target = np.ones_like(per_example)
        for name in ("k", "c"):
            if name in fixed:
                target = target - fixed[name] * columns[name]
        free = [name for name in ("k", "c") if name not in fixed]
        params = {name: float(fixed[name]) for name in ("k", "c") if name in fixed}
        if free:
            solution = lsq_linear(
                np.column_stack([columns[name] for name in free]), target, bounds=(0.0, np.inf)
            )
            params.update({name: float(v) for name, v in zip(free, solution.x, strict=True)})
            residual = float(np.sum((np.column_stack([columns[f] for f in free]) @ solution.x - target) ** 2))
        else:
            residual = float(np.sum(target**2))
        return {**params, "a": a}, residual

    @staticmethod
    def fit(n: np.ndarray, d: np.ndarray, c: np.ndarray, fixed: Mapping[str, float]) -> dict[str, float]:
        """Variable projection: ``k`` and ``c`` are linear given ``a``, which is searched on its own."""
        per_example = c / d
        if "a" in fixed:
            params, _ = _PerExampleCost._solve_linear(n, per_example, float(fixed["a"]), fixed)
            return params
        if np.ptp(np.log(n)) == 0:
            raise CostError("The recorded runs cannot identify 'a': they all have the same model size")
        result = minimize_scalar(
            lambda a: _PerExampleCost._solve_linear(n, per_example, a, fixed)[1],
            bounds=EXPONENT_RANGE,
            method="bounded",
        )
        params, _ = _PerExampleCost._solve_linear(n, per_example, float(result.x), fixed)
        return params


_FORMS: dict[str, Any] = {form.name: form for form in (_PowerCost, _PerExampleCost)}


def available_cost_forms() -> dict[str, str]:
    """Return a mapping from cost-form name to a one-line description.

    Examples:
        >>> for name, description in available_cost_forms().items():
        ...     print(f"{name}: {description}")
        power: k * N**a * D**b (e.g. k=6, a=b=1: training FLOPs of a transformer)
        per-example: D * (k * N**a + c) (per-example cost growing with N, plus a fixed overhead)
    """
    return {name: form.description for name, form in _FORMS.items()}


def available_cost_params() -> dict[str, tuple[str, ...]]:
    """Return a mapping from cost-form name to its parameter names.

    Examples:
        >>> available_cost_params()["per-example"]
        ('k', 'a', 'c')
    """
    return {name: form.param_names for name, form in _FORMS.items()}


def _form(name: str) -> Any:
    if name not in _FORMS:
        raise CostError(f"Unknown cost form {name!r}; available: {', '.join(_FORMS)}")
    return _FORMS[name]


def _validate(form: str, params: Mapping[str, float]) -> dict[str, float]:
    """Check a complete parameter set: the cost must be positive and grow with both resources."""
    spec = _form(form)
    unknown = set(params) - set(spec.param_names)
    if unknown:
        raise CostError(
            f"Cost form {form!r} has no parameter(s) {sorted(unknown)}; its are {spec.param_names}"
        )
    values = {name: float(params[name]) for name in spec.param_names if name in params}
    if not all(np.isfinite(v) for v in values.values()):
        raise CostError(f"Cost parameters must be finite, got {values}")
    for name, value in values.items():
        if name == "c":
            if value < 0:
                raise CostError(f"The overhead 'c' cannot be negative, got {value}")
        elif value <= 0:
            raise CostError(
                f"Cost parameter {name!r} must be positive -- a run cannot get cheaper as it grows -- "
                f"got {value}"
            )
    return values


@dataclasses.dataclass(frozen=True)
class CostModel:
    """What a training run of model size ``N`` and dataset size ``D`` costs.

    Build one with :func:`cost_model` (declared) or :func:`fit_cost_model` (from recorded runs).

    Attributes:
        form: The cost form, see :func:`available_cost_forms`.
        params: Every parameter of the form.
        fitted: Names of the parameters that were estimated from records rather than declared.
        log_rmse: For a fitted model, the root-mean-square error of its predictions on the log scale,
            over the runs it was fitted to; ``None`` for a declared one.
        n_runs: Number of runs the model was fitted to, or ``None``.
        notes: Objections raised while fitting it.
    """

    form: str
    params: dict[str, float]
    fitted: tuple[str, ...] = ()
    log_rmse: float | None = None
    n_runs: int | None = None
    notes: tuple[Note, ...] = ()

    def __call__(self, model_size: Any, dataset_size: Any) -> np.ndarray:
        """The cost of training at these model and dataset sizes."""
        n, d = np.asarray(model_size, dtype=float), np.asarray(dataset_size, dtype=float)
        return _form(self.form).evaluate(self.params, n, d)

    def dataset_size_for(self, budget: Any, model_size: Any) -> np.ndarray:
        """The dataset size that spends ``budget`` exactly at this model size: the iso-cost curve.

        Examples:
            >>> overhead = cost_model("per-example", k=6.0, a=1.0, c=6e8)
            >>> f"{float(overhead.dataset_size_for(1.2e20, 1e8)):.3g}"
            '1e+11'
        """
        budget, n = np.asarray(budget, dtype=float), np.asarray(model_size, dtype=float)
        return _form(self.form).dataset_size_for(self.params, budget, n)

    @property
    def description(self) -> str:
        """The form with its parameter values substituted, for summaries.

        Examples:
            >>> cost_model("power", k=6.0, a=1.0, b=1.0).description
            'power: k=6, a=1, b=1'
        """
        values = ", ".join(f"{name}={self.params[name]:.4g}" for name in _form(self.form).param_names)
        return f"{self.form}: {values}"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe description that :meth:`from_dict` reverses."""
        return {
            "form": self.form,
            "params": dict(self.params),
            "fitted": list(self.fitted),
            "log_rmse": self.log_rmse,
            "n_runs": self.n_runs,
            "notes": [note.to_dict() for note in self.notes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostModel:
        """Rebuild from :meth:`to_dict` output.

        Examples:
            >>> declared = cost_model("power", k=6.0, a=1.0, b=1.0)
            >>> CostModel.from_dict(declared.to_dict()) == declared
            True
        """
        return cls(
            form=data["form"],
            params=_validate(data["form"], data["params"]),
            fitted=tuple(data.get("fitted", ())),
            log_rmse=data.get("log_rmse"),
            n_runs=data.get("n_runs"),
            notes=tuple(Note.from_dict(note) for note in data.get("notes", ())),
        )


def cost_model(form: str = "power", **params: float) -> CostModel:
    """Declare a cost model with every parameter given.

    Args:
        form: The cost form, see :func:`available_cost_forms`.
        **params: A value for every parameter of the form.

    Returns:
        The :class:`CostModel`.

    Raises:
        CostError: If the form is unknown, or a parameter is missing or invalid. A form with some
            parameters unknown is fitted with :func:`fit_cost_model`, passing the known ones as
            ``fixed``.

    Examples:
        >>> cost_model("power", k=6.0, a=1.0)
        Traceback (most recent call last):
        ...
        simple_scaling_laws.cost.CostError: Cost form 'power' needs ['b'] as well; to estimate them from
        recorded runs, use fit_cost_model(..., form='power', fixed={...})
    """
    values = _validate(form, params)
    missing = [name for name in _form(form).param_names if name not in values]
    if missing:
        raise CostError(
            f"Cost form {form!r} needs {missing} as well; to estimate them from recorded runs, use "
            f"fit_cost_model(..., form={form!r}, fixed={{...}})"
        )
    return CostModel(form=form, params=values)


def _compute_column(columns: list[str], compute: str | None) -> str:
    if compute is not None:
        if compute not in columns:
            raise CostError(f"Compute column {compute!r} is not in the table")
        return compute
    found = [c for c in columns if c.startswith(f"{COMPUTE_ROLE}{SEP}")]
    if len(found) != 1:
        raise CostError(
            f"Expected exactly one column prefixed '{COMPUTE_ROLE}{SEP}' holding each run's recorded cost, "
            f"found {found}; name it with compute=..."
        )
    return found[0]


def run_costs(
    source: Any, columns: Mapping[str, Any] | None = None, compute: str | None = None
) -> tuple[pl.DataFrame, list[Note]]:
    """One row per training run: its model size, dataset size and recorded cost.

    Args:
        source: A parquet/CSV path or a dataframe of evaluation rows (or one row per run).
        columns: Optional column-role overrides, as for :func:`simple_scaling_laws.fit`.
        compute: The recorded-cost column; by default the single ``compute__*`` column.

    Returns:
        The per-run table, with columns ``training_run_id``, ``model_size``, ``dataset_size`` and
        ``compute``, and any notes raised while building it.

    Raises:
        CostError: If the table does not have exactly one predictor of each role, or no usable costs.
    """
    frame = read_table(source)
    overrides = dict(columns or {})
    run = overrides.get(ID_ROLES[0], ID_ROLES[0])
    if run not in frame.columns:
        raise CostError(f"No training-run ID column {run!r}")
    found = {}
    for role in PREDICTOR_ROLES:
        named = overrides.get(role)
        if named is not None:
            found[role] = [named] if isinstance(named, str) else list(named)
        else:
            found[role] = [c for c in frame.columns if c.startswith(f"{role}{SEP}")]
    if len(found["model_size"]) != 1 or len(found["dataset_size"]) != 1:
        raise CostError(
            "A cost model is over exactly one model-size and one dataset-size column, got "
            f"{found['model_size']} and {found['dataset_size']}"
        )
    column = _compute_column(frame.columns, compute)
    valid = frame.select(
        pl.col(run).cast(pl.String).alias("training_run_id"),
        pl.col(found["model_size"][0]).cast(pl.Float64).alias("model_size"),
        pl.col(found["dataset_size"][0]).cast(pl.Float64).alias("dataset_size"),
        pl.col(column).cast(pl.Float64, strict=False).alias("compute"),
    ).filter(pl.col("compute").is_finite() & (pl.col("compute") > 0))
    if valid.height == 0:
        raise CostError(f"Column {column!r} has no positive, finite recorded costs")
    runs = valid.group_by("training_run_id", maintain_order=True).agg(
        pl.col("model_size").first(),
        pl.col("dataset_size").first(),
        pl.col("compute").mean(),
        (pl.col("compute").max() / pl.col("compute").min() - 1.0).alias("_spread"),
    )
    notes = []
    inconsistent = runs.filter(pl.col("_spread") > WITHIN_RUN_TOLERANCE)
    if inconsistent.height:
        notes.append(
            Note(
                "cost_varies_within_run",
                "warning",
                f"{inconsistent.height} run(s) record different costs on different rows of {column!r}; "
                "a run has one cost, so their mean was used. If the column is per evaluation rather "
                "than per training run, it is the wrong column for a cost model.",
                {"column": column, "n_runs": inconsistent.height},
            )
        )
    return runs.drop("_spread"), notes


def fit_cost_model(
    source: Any,
    form: str = "power",
    fixed: Mapping[str, float] | None = None,
    columns: Mapping[str, Any] | None = None,
    compute: str | None = None,
) -> CostModel:
    """Fit a cost model to runs that recorded what they cost.

    The fit is least squares on the log of the cost (relative error, for ``per-example``), so a run
    costing a thousand times more does not dominate it. Parameters given in ``fixed`` are held at
    their values; the rest are estimated.

    Args:
        source: A parquet/CSV path or a dataframe with a training-run ID, one model-size and one
            dataset-size column, and a recorded-cost column.
        form: The cost form, see :func:`available_cost_forms`.
        fixed: Known parameter values, e.g. ``{"b": 1.0}`` when cost is known to be linear in data.
        columns: Optional column-role overrides, as for :func:`simple_scaling_laws.fit`.
        compute: The recorded-cost column; by default the single ``compute__*`` column.

    Returns:
        The fitted :class:`CostModel`, with its log-scale error and any notes.

    Raises:
        CostError: If the records cannot identify the free parameters.

    Examples:
        >>> runs = pl.DataFrame(
        ...     {
        ...         "training_run_id": ["a", "b", "c", "d"],
        ...         "model_size__n_params": [1e6, 1e7, 1e6, 1e7],
        ...         "dataset_size__n_tokens": [1e8, 1e8, 1e9, 1e9],
        ...         "compute__flops": [6e14, 6e15, 6e15, 6e16],
        ...     }
        ... )
        >>> cost = fit_cost_model(runs, form="power")
        >>> {name: round(value, 3) for name, value in cost.params.items()}
        {'k': 6.0, 'a': 1.0, 'b': 1.0}
        >>> cost.fitted, round(cost.log_rmse, 6)
        (('k', 'a', 'b'), 0.0)
    """
    spec = _form(form)
    fixed_values = _validate(form, fixed or {})
    runs, notes = run_costs(source, columns, compute)
    free = [name for name in spec.param_names if name not in fixed_values]
    if not free:
        raise CostError("Every parameter is fixed, so there is nothing to fit; use cost_model instead")
    if runs.height < len(free) + 1:
        raise CostError(
            f"{runs.height} run(s) with a recorded cost cannot fit {len(free)} parameter(s) and judge "
            f"the fit; need at least {len(free) + 1}"
        )
    n, d, c = (runs[name].to_numpy() for name in ("model_size", "dataset_size", "compute"))
    params = _validate(form, spec.fit(n, d, c, fixed_values))
    error = float(np.sqrt(np.mean(np.log(spec.evaluate(params, n, d) / c) ** 2)))
    if error > POOR_FIT_LOG_ERROR:
        notes.append(
            Note(
                "cost_model_poor_fit",
                "warning",
                f"The {form!r} cost model misses the recorded costs by {np.expm1(error):.0%} on a typical "
                "run. Allocations built on it inherit that error; another form, or a cost column that "
                "measures the same thing in every run, may fit better.",
                {"form": form, "log_rmse": error},
            )
        )
    return CostModel(
        form=form, params=params, fitted=tuple(free), log_rmse=error, n_runs=runs.height, notes=tuple(notes)
    )
