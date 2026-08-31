"""What a target column *is*: which way is better, and what range it lives in.

Two facts about a target cannot be read off its values, and both change the answer:

* **Direction.** AUROC is better when larger; Brier score and calibration error are better when
  smaller. Both are ``test_metric__*`` columns, so the role prefix cannot tell them apart. Direction
  is what turns a difference between two systems into a win rate.
* **Support.** A metric confined to ``[0, 1]`` fit on its raw scale can produce an asymptote outside
  ``[0, 1]``, which is not a meaningful statement about a classifier. Fitting on the logit scale
  makes the bound hold by construction.

Both are looked up in a small table of metrics people actually use, keyed on the part of the column
name after the role prefix. Anything not in the table is treated as an ordinary unbounded quantity,
and either fact can be stated explicitly for a target the table does not know.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

#: Supports a target may have. ``unit`` means the value is confined to ``[0, 1]``.
REAL, UNIT = "real", "unit"

#: Directions a target may improve in.
HIGHER, LOWER = "higher", "lower"

#: Where a target's description came from, which is what tells a caller how much to trust it.
REGISTRY, OVERRIDE, DEFAULT = "registry", "override", "default"

#: Values are pulled this far inside the open interval before the logit, since a metric that
#: saturates at exactly 0 or 1 -- an AUROC of 1.0 on a small test set, say -- has no finite logit.
UNIT_EPSILON = 1e-6

#: Metrics whose direction and support are known. Keyed on the column name after its role prefix,
#: lowercased. Deliberately short: it covers what is actually reported, and anything missing can be
#: declared explicitly rather than guessed at from a fuzzy name match.
KNOWN_METRICS: dict[str, tuple[str, str]] = {
    # Ranking and classification quality: larger is better, confined to [0, 1].
    "auroc": (UNIT, HIGHER),
    "auc": (UNIT, HIGHER),
    "roc_auc": (UNIT, HIGHER),
    "auprc": (UNIT, HIGHER),
    "pr_auc": (UNIT, HIGHER),
    "average_precision": (UNIT, HIGHER),
    "accuracy": (UNIT, HIGHER),
    "balanced_accuracy": (UNIT, HIGHER),
    "f1": (UNIT, HIGHER),
    "precision": (UNIT, HIGHER),
    "recall": (UNIT, HIGHER),
    "specificity": (UNIT, HIGHER),
    "sensitivity": (UNIT, HIGHER),
    # Error and calibration: smaller is better, confined to [0, 1].
    "brier": (UNIT, LOWER),
    "brier_score": (UNIT, LOWER),
    "ece": (UNIT, LOWER),
    "mce": (UNIT, LOWER),
    "calibration_error": (UNIT, LOWER),
    "error_rate": (UNIT, LOWER),
    # Unbounded, smaller is better.
    "perplexity": (REAL, LOWER),
    "mse": (REAL, LOWER),
    "mae": (REAL, LOWER),
    "rmse": (REAL, LOWER),
}

#: Keys accepted when describing a target explicitly.
OVERRIDE_KEYS: frozenset[str] = frozenset({"support", "direction"})


class MetricError(ValueError):
    """Raised when a target description is malformed or contradicts the data."""


@dataclasses.dataclass(frozen=True, slots=True)
class MetricKind:
    """How a target behaves.

    Attributes:
        support: :data:`REAL` or :data:`UNIT`.
        direction: :data:`HIGHER` or :data:`LOWER`, whichever counts as better.
        source: :data:`REGISTRY`, :data:`OVERRIDE` or :data:`DEFAULT`. A ``DEFAULT`` direction is an
            assumption, not knowledge, and callers that act on it should say so.

    Examples:
        >>> MetricKind(UNIT, HIGHER, REGISTRY).transforms
        True
        >>> MetricKind(REAL, LOWER, DEFAULT).transforms
        False
    """

    support: str
    direction: str
    source: str

    @property
    def transforms(self) -> bool:
        """Whether values are fit on a transformed scale."""
        return self.support == UNIT

    @property
    def known_direction(self) -> bool:
        """Whether the direction was stated or looked up rather than assumed."""
        return self.source != DEFAULT

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-compatible dictionary."""
        return {"support": self.support, "direction": self.direction, "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> MetricKind:
        """Rebuild from :meth:`to_dict` output."""
        return cls(data["support"], data["direction"], data.get("source", DEFAULT))


def describe(short_name: str, is_loss: bool, override: Mapping[str, str] | None = None) -> MetricKind:
    """Determine a target's support and direction.

    Args:
        short_name: The column name with its role prefix stripped, e.g. ``"auroc"``.
        is_loss: Whether the column is a loss, which fixes the direction as lower-is-better.
        override: An explicit ``{"support": ..., "direction": ...}`` description, which wins.

    Returns:
        The resolved :class:`MetricKind`.

    Raises:
        MetricError: If the override contains unknown keys or values.

    Examples:
        >>> describe("auroc", is_loss=False)
        MetricKind(support='unit', direction='higher', source='registry')
        >>> describe("brier", is_loss=False)
        MetricKind(support='unit', direction='lower', source='registry')
        >>> describe("cross_entropy", is_loss=True)
        MetricKind(support='real', direction='lower', source='registry')

        An unrecognized metric is assumed unbounded and larger-is-better, and says that the
        direction was assumed:

        >>> describe("some_score", is_loss=False)
        MetricKind(support='real', direction='higher', source='default')

        Anything can be declared explicitly:

        >>> describe("some_score", is_loss=False, override={"support": "unit"})
        MetricKind(support='unit', direction='higher', source='override')
        >>> describe("some_score", is_loss=False, override={"whoops": "unit"})
        Traceback (most recent call last):
        ...
        simple_scaling_laws.metrics.MetricError: Unknown target description key(s) ['whoops'];
        allowed: ['direction', 'support']
    """
    override = dict(override or {})
    unknown = set(override) - OVERRIDE_KEYS
    if unknown:
        raise MetricError(
            f"Unknown target description key(s) {sorted(unknown)}; allowed: {sorted(OVERRIDE_KEYS)}"
        )
    for key, allowed in (("support", (REAL, UNIT)), ("direction", (HIGHER, LOWER))):
        if key in override and override[key] not in allowed:
            raise MetricError(f"{key} must be one of {list(allowed)}, got {override[key]!r}")

    registered = KNOWN_METRICS.get(short_name.lower())
    if registered is not None:
        support, direction, source = *registered, REGISTRY
    elif is_loss:
        support, direction, source = REAL, LOWER, REGISTRY
    else:
        support, direction, source = REAL, HIGHER, DEFAULT
    if override:
        support = override.get("support", support)
        direction = override.get("direction", direction)
        source = OVERRIDE
    return MetricKind(support=support, direction=direction, source=source)


def to_fitting_scale(values: np.ndarray, kind: MetricKind) -> np.ndarray:
    """Map observed values onto the scale the law is fit on.

    A ``unit``-supported target is fit on its logit. That is what keeps a fitted asymptote inside
    ``[0, 1]``: the law is unconstrained on the logit scale, and every value it can take maps back
    into the open unit interval. It also makes the constant-variance assumption far more plausible,
    since an AUROC near 0.99 has much less room to move than one near 0.7.

    Args:
        values: Observed values.
        kind: The target's description.

    Returns:
        Values on the fitting scale.

    Raises:
        MetricError: If a ``unit`` target has values outside ``[0, 1]``.

    Examples:
        >>> to_fitting_scale(np.array([0.5, 0.75]), MetricKind(UNIT, HIGHER, REGISTRY)).round(4)
        array([0.    , 1.0986])
        >>> to_fitting_scale(np.array([2.0, 3.0]), MetricKind(REAL, LOWER, REGISTRY))
        array([2., 3.])
    """
    values = np.asarray(values, dtype=float)
    if not kind.transforms:
        return values
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise MetricError(
            f"Target is declared to lie in [0, 1] but its values span "
            f"[{finite.min():.4g}, {finite.max():.4g}]. Correct the values, or describe the target "
            'as unbounded with {"support": "real"}.'
        )
    clipped = np.clip(values, UNIT_EPSILON, 1.0 - UNIT_EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def from_fitting_scale(values: np.ndarray, kind: MetricKind) -> np.ndarray:
    """Map values on the fitting scale back to the target's own scale.

    The inverse of :func:`to_fitting_scale`. It is monotone, so quantiles map straight through: the
    2.5th percentile of the back-transformed draws is the back-transformed 2.5th percentile.

    Args:
        values: Values on the fitting scale.
        kind: The target's description.

    Returns:
        Values on the target's own scale.

    Examples:
        >>> from_fitting_scale(np.array([0.0, 1.0986]), MetricKind(UNIT, HIGHER, REGISTRY)).round(4)
        array([0.5 , 0.75])

        Round-tripping is exact well inside the interval:

        >>> kind = MetricKind(UNIT, HIGHER, REGISTRY)
        >>> original = np.array([0.05, 0.5, 0.99])
        >>> bool(np.allclose(from_fitting_scale(to_fitting_scale(original, kind), kind), original))
        True
    """
    values = np.asarray(values, dtype=float)
    if not kind.transforms:
        return values
    # Written as the numerically stable form of the logistic, so a large draw saturates at 1.0
    # rather than overflowing on the way there.
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def scale_description(kind: MetricKind) -> str:
    """A short human-readable name for the fitting scale, for summaries and diagnostics."""
    return "logit" if kind.transforms else "identity"


def parse_overrides(targets: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Validate a mapping of target column name to its explicit description.

    Args:
        targets: For example ``{"test_metric__custom": {"support": "unit", "direction": "lower"}}``.

    Returns:
        The same mapping, validated.

    Raises:
        MetricError: If a description is not a mapping or contains unknown keys or values.

    Examples:
        >>> parse_overrides({"test_metric__x": {"support": "unit"}})
        {'test_metric__x': {'support': 'unit'}}
        >>> parse_overrides(None)
        {}
        >>> parse_overrides({"test_metric__x": "unit"})
        Traceback (most recent call last):
        ...
        simple_scaling_laws.metrics.MetricError: Description of 'test_metric__x' must be a mapping
        like {'support': 'unit', 'direction': 'higher'}, got str
    """
    out: dict[str, dict[str, str]] = {}
    for name, description in dict(targets or {}).items():
        if not isinstance(description, dict):
            raise MetricError(
                f"Description of {name!r} must be a mapping like "
                "{'support': 'unit', 'direction': 'higher'}, got " + type(description).__name__
            )
        describe(name, is_loss=False, override=description)  # validation only
        out[name] = dict(description)
    return out
