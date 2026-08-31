"""Discovery and validation of column roles in a scaling-law input table.

The package is designed to need *no* schema configuration when the conventional ``role__name`` prefixes are
used. This module turns a list of column names into a :class:`Schema` describing which columns are
identifiers, which are scale predictors, and which are fitting targets.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .metrics import MetricKind, describe, parse_overrides

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

#: Reserved identifier columns. Only ``training_run_id`` is required.
ID_ROLES: tuple[str, ...] = ("training_run_id", "train_set_id", "test_set_id", "optimizer_seed")

#: Column-name prefixes that identify predictor columns, in canonical order.
PREDICTOR_ROLES: tuple[str, ...] = ("model_size", "dataset_size")

#: Column-name prefixes that identify fitting targets, in priority order. The first role with any
#: matching column supplies the primary target unless one is named explicitly.
TARGET_ROLES: tuple[str, ...] = ("test_loss", "train_loss", "test_metric", "train_metric")

#: Target roles whose values are losses (lower is better, amplitudes constrained non-negative).
LOSS_ROLES: frozenset[str] = frozenset({"test_loss", "train_loss"})

#: Separator between a role prefix and the rest of a column name.
SEP = "__"


class SchemaError(ValueError):
    """Raised when column roles cannot be resolved into a usable schema."""


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """One fitting target column.

    Attributes:
        name: The column name.
        role: One of :data:`TARGET_ROLES`.
        kind: How the target behaves -- which direction is better, and whether it is confined to
            ``[0, 1]`` and therefore fit on a transformed scale.

    Examples:
        >>> Target("test_loss__cross_entropy", "test_loss").is_loss
        True
        >>> Target("test_metric__auroc", "test_metric").signed_amplitude
        True

        A target's kind is looked up from its short name unless stated explicitly:

        >>> Target.build("test_metric__auroc", "test_metric").kind.support
        'unit'
        >>> Target.build("test_loss__cross_entropy", "test_loss").kind.direction
        'lower'
    """

    name: str
    role: str
    kind: MetricKind = dataclasses.field(default_factory=lambda: MetricKind("real", "lower", "default"))

    @classmethod
    def build(cls, name: str, role: str, override: Mapping[str, str] | None = None) -> Target:
        """Construct a target, resolving its kind from the registry or an explicit description.

        Args:
            name: The column name.
            role: One of :data:`TARGET_ROLES`.
            override: An explicit ``{"support": ..., "direction": ...}`` description.

        Returns:
            The target.
        """
        prefix = f"{role}{SEP}"
        short = name[len(prefix) :] if name.startswith(prefix) else name
        return cls(name=name, role=role, kind=describe(short, role in LOSS_ROLES, override))

    @property
    def is_loss(self) -> bool:
        """Whether this target is a loss (lower is better)."""
        return self.role in LOSS_ROLES

    @property
    def signed_amplitude(self) -> bool:
        """Whether the law's amplitudes may be negative for this target.

        Losses use the conventional asymptotic-floor parameterization with non-negative amplitudes. Metrics
        may increase with scale, so their amplitudes are free in sign and the curve can approach its asymptote
        from below.
        """
        return not self.is_loss

    @property
    def short_name(self) -> str:
        """The target name with its role prefix stripped.

        Examples:
            >>> Target("test_metric__auroc", "test_metric").short_name
            'auroc'
        """
        prefix = f"{self.role}{SEP}"
        return self.name[len(prefix) :] if self.name.startswith(prefix) else self.name


@dataclasses.dataclass(frozen=True, slots=True)
class Schema:
    """Resolved column roles for a scaling-law input table.

    Attributes:
        training_run_id: Column identifying one independently trained model.
        model_size: Columns describing model scale, in table order.
        dataset_size: Columns describing dataset scale, in table order.
        targets: All target columns that will be fit.
        primary_target: The target used as the reference for cross-metric diagnostics.
        train_set_id: Optional column identifying the training split.
        test_set_id: Optional column identifying an evaluation resample.
        optimizer_seed: Optional column identifying the optimizer seed.
    """

    training_run_id: str
    model_size: tuple[str, ...]
    dataset_size: tuple[str, ...]
    targets: tuple[Target, ...]
    primary_target: str
    train_set_id: str | None = None
    test_set_id: str | None = None
    optimizer_seed: str | None = None

    @property
    def predictors(self) -> tuple[str, ...]:
        """All predictor columns, model-size columns first."""
        return self.model_size + self.dataset_size

    @property
    def target_names(self) -> tuple[str, ...]:
        """Names of all target columns."""
        return tuple(t.name for t in self.targets)

    def target(self, name: str) -> Target:
        """Look up a target by column name.

        Args:
            name: The target column name.

        Returns:
            The matching :class:`Target`.

        Raises:
            KeyError: If no such target exists in this schema.
        """
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(f"Unknown target {name!r}; known targets are {list(self.target_names)}")

    def predictor_role(self, name: str) -> str:
        """Return ``'model_size'`` or ``'dataset_size'`` for a predictor column."""
        if name in self.model_size:
            return "model_size"
        if name in self.dataset_size:
            return "dataset_size"
        raise KeyError(f"Unknown predictor {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "training_run_id": self.training_run_id,
            "train_set_id": self.train_set_id,
            "test_set_id": self.test_set_id,
            "optimizer_seed": self.optimizer_seed,
            "model_size": list(self.model_size),
            "dataset_size": list(self.dataset_size),
            "targets": [{"name": t.name, "role": t.role, "kind": t.kind.to_dict()} for t in self.targets],
            "primary_target": self.primary_target,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Schema:
        """Rebuild a schema from :meth:`to_dict` output.

        Examples:
            >>> schema = discover_schema(
            ...     ["training_run_id", "model_size__n", "dataset_size__d", "test_loss__ce"]
            ... )
            >>> Schema.from_dict(schema.to_dict()) == schema
            True
        """
        return cls(
            training_run_id=data["training_run_id"],
            train_set_id=data.get("train_set_id"),
            test_set_id=data.get("test_set_id"),
            optimizer_seed=data.get("optimizer_seed"),
            model_size=tuple(data["model_size"]),
            dataset_size=tuple(data["dataset_size"]),
            targets=tuple(
                Target(t["name"], t["role"], MetricKind.from_dict(t["kind"])) for t in data["targets"]
            ),
            primary_target=data["primary_target"],
        )


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a string-or-iterable override into a tuple of column names."""
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _check_present(names: Iterable[str], columns: Sequence[str], what: str) -> None:
    """Raise :class:`SchemaError` if any of ``names`` is missing from ``columns``."""
    missing = [n for n in names if n not in columns]
    if missing:
        raise SchemaError(f"{what} column(s) {missing} not found in table columns {list(columns)}")


def discover_schema(
    columns: Sequence[str],
    overrides: Mapping[str, str | Iterable[str]] | None = None,
    primary_target: str | None = None,
    targets: Mapping[str, Any] | None = None,
) -> Schema:
    """Infer column roles from column names, with optional explicit overrides.

    Columns are assigned by prefix: ``model_size__*`` and ``dataset_size__*`` are predictors;
    ``test_loss__*``, ``train_loss__*``, ``test_metric__*`` and ``train_metric__*`` are targets.
    Any column not matching a prefix and not a reserved identifier is ignored, so extra bookkeeping
    columns are harmless.

    Args:
        columns: The column names of the input table.
        overrides: Optional map from a role name to the column (or columns) filling it. Identifier
            roles take a single column name; ``model_size``, ``dataset_size`` and the target roles
            take a column name or a list of them.
        primary_target: Optional explicit primary target column name.
        targets: Optional explicit per-target descriptions, e.g.
            ``{"test_metric__custom": {"support": "unit", "direction": "lower"}}``. Anything not
            named here is looked up in :data:`simple_scaling_laws.metrics.KNOWN_METRICS`.

    Returns:
        The resolved :class:`Schema`.

    Raises:
        SchemaError: If required roles cannot be resolved.

    Examples:
        >>> schema = discover_schema(
        ...     [
        ...         "training_run_id",
        ...         "test_set_id",
        ...         "optimizer_seed",
        ...         "model_size__n_params",
        ...         "dataset_size__n_subjects",
        ...         "test_loss__cross_entropy",
        ...         "test_metric__auroc",
        ...         "notes",
        ...     ]
        ... )
        >>> schema.predictors
        ('model_size__n_params', 'dataset_size__n_subjects')
        >>> schema.target_names
        ('test_loss__cross_entropy', 'test_metric__auroc')
        >>> schema.primary_target
        'test_loss__cross_entropy'
        >>> schema.optimizer_seed
        'optimizer_seed'

        Overrides accept non-conventional names:

        >>> schema = discover_schema(
        ...     ["run", "boot", "params", "n_train", "loss"],
        ...     overrides={
        ...         "training_run_id": "run",
        ...         "test_set_id": "boot",
        ...         "model_size": "params",
        ...         "dataset_size": ["n_train"],
        ...         "test_loss": "loss",
        ...     },
        ... )
        >>> schema.training_run_id, schema.primary_target
        ('run', 'loss')

        Unresolvable tables fail loudly:

        >>> discover_schema(["training_run_id", "model_size__n"])
        Traceback (most recent call last):
        ...
        simple_scaling_laws.schema.SchemaError: No target columns found. Expected at least one
        column prefixed with one of: test_loss__, train_loss__, test_metric__, train_metric__.
    """
    overrides = dict(overrides or {})
    descriptions = parse_overrides(targets)
    unknown_roles = set(overrides) - set(ID_ROLES) - set(PREDICTOR_ROLES) - set(TARGET_ROLES)
    if unknown_roles:
        known = list(ID_ROLES) + list(PREDICTOR_ROLES) + list(TARGET_ROLES)
        raise SchemaError(f"Unknown column override role(s) {sorted(unknown_roles)}; known roles: {known}")

    ids: dict[str, str | None] = {}
    for role in ID_ROLES:
        if role in overrides:
            name = _as_tuple(overrides[role])
            if len(name) != 1:
                raise SchemaError(f"Override for {role!r} must name exactly one column, got {list(name)}")
            _check_present(name, columns, role)
            ids[role] = name[0]
        else:
            ids[role] = role if role in columns else None
    if ids["training_run_id"] is None:
        raise SchemaError(
            "No 'training_run_id' column found. Supply one, or map it with "
            "columns={'training_run_id': '<your column>'}."
        )

    predictors: dict[str, tuple[str, ...]] = {}
    for role in PREDICTOR_ROLES:
        if role in overrides:
            found = _as_tuple(overrides[role])
            _check_present(found, columns, role)
        else:
            found = tuple(c for c in columns if c.startswith(f"{role}{SEP}"))
        predictors[role] = found
    if not predictors["model_size"] and not predictors["dataset_size"]:
        raise SchemaError(
            "No predictor columns found. Expected at least one column prefixed with "
            f"'model_size{SEP}' or 'dataset_size{SEP}'."
        )

    targets: list[Target] = []
    for role in TARGET_ROLES:
        if role in overrides:
            found = _as_tuple(overrides[role])
            _check_present(found, columns, role)
        else:
            found = tuple(c for c in columns if c.startswith(f"{role}{SEP}"))
        targets.extend(Target.build(name, role, descriptions.get(name)) for name in found)
    if not targets:
        prefixes = ", ".join(f"{r}{SEP}" for r in TARGET_ROLES)
        raise SchemaError(
            f"No target columns found. Expected at least one column prefixed with one of: {prefixes}."
        )

    unknown_targets = set(descriptions) - {t.name for t in targets}
    if unknown_targets:
        raise SchemaError(
            f"Description(s) given for {sorted(unknown_targets)}, which are not target columns. "
            f"Known targets: {sorted(t.name for t in targets)}"
        )

    seen: dict[str, str] = {}
    for t in targets:
        if t.name in seen:
            raise SchemaError(f"Column {t.name!r} is claimed by both roles {seen[t.name]!r} and {t.role!r}")
        seen[t.name] = t.role

    overlap = set(seen) & set(predictors["model_size"] + predictors["dataset_size"])
    if overlap:
        raise SchemaError(f"Column(s) {sorted(overlap)} are used as both predictors and targets")

    if primary_target is None:
        primary_target = _default_primary_target(targets)
    elif primary_target not in seen:
        raise SchemaError(f"primary_target {primary_target!r} is not one of the targets {sorted(seen)}")

    return Schema(
        training_run_id=ids["training_run_id"],
        train_set_id=ids["train_set_id"],
        test_set_id=ids["test_set_id"],
        optimizer_seed=ids["optimizer_seed"],
        model_size=predictors["model_size"],
        dataset_size=predictors["dataset_size"],
        targets=tuple(targets),
        primary_target=primary_target,
    )


def _default_primary_target(targets: Sequence[Target]) -> str:
    """Choose the primary target: the first target of the highest-priority available role.

    Examples:
        >>> _default_primary_target(
        ...     [Target("test_metric__auroc", "test_metric"), Target("test_loss__ce", "test_loss")]
        ... )
        'test_loss__ce'
    """
    for role in TARGET_ROLES:
        candidates = [t.name for t in targets if t.role == role]
        if candidates:
            return candidates[0]
    raise SchemaError("No targets available")  # pragma: no cover - guarded by caller
