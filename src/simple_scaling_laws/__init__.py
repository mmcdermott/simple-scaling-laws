"""Fit empirical ML scaling laws, with conservative uncertainty, from repeated experiment records.

The package answers one question: *given a handful of trained models at different scales, what
should we expect at a larger scale, and how sure are we?* It is deliberately opinionated -- the
statistics are package policy, not user configuration.

Examples:
    >>> from simple_scaling_laws import fit
    >>> from simple_scaling_laws.simulate import simulate_runs
    >>> frame = simulate_runs(
    ...     {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}},
    ...     run_sd=0.01,
    ...     eval_sd=0.02,
    ...     seed=0,
    ... )
    >>> model = fit(frame, n_draws=200, seed=0)
    >>> round(model.params("test_loss__cross_entropy")["alpha"], 1)
    0.3
"""

from .api import fit, load_config
from .artifact import FORMAT_VERSION
from .compare import compare
from .data import build_dataset
from .laws import available_laws
from .model import ScalingLawModel
from .notes import Note
from .schema import Schema, discover_schema

__all__ = [
    "FORMAT_VERSION",
    "Note",
    "ScalingLawModel",
    "Schema",
    "available_laws",
    "build_dataset",
    "compare",
    "discover_schema",
    "fit",
    "load",
    "load_config",
]


def load(path):
    """Load a fitted model from a ``.slaw`` artifact directory.

    Args:
        path: The artifact directory.

    Returns:
        The reconstructed :class:`~simple_scaling_laws.model.ScalingLawModel`.
    """
    return ScalingLawModel.load(path)
