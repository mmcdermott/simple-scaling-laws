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

    Every error the package raises on bad input is a ``ValueError`` subclass exported here, so a
    caller can catch the specific one:

    >>> from simple_scaling_laws import LawError
    >>> try:
    ...     fit(frame, law="no-such-law")
    ... except LawError as error:
    ...     print(type(error).__name__)
    LawError
"""

from .api import ConfigError, fit, load_config
from .artifact import FORMAT_VERSION, ArtifactError, package_version
from .compare import ComparisonError, compare
from .cost import CostError, CostModel, available_cost_forms, cost_model, fit_cost_model
from .data import DataError, build_dataset
from .design import isoflop_grid, score_design, suggest_runs
from .isoflop import IsoflopError, IsoflopFit, fit_isoflop
from .laws import LawError, available_laws
from .metrics import MetricError
from .model import PredictionError, ScalingLawModel
from .notes import Note
from .schema import Schema, SchemaError, discover_schema

#: The installed version, from the package metadata (set by the release tag).
__version__ = package_version()

__all__ = [
    "FORMAT_VERSION",
    "ArtifactError",
    "ComparisonError",
    "ConfigError",
    "CostError",
    "CostModel",
    "DataError",
    "IsoflopError",
    "IsoflopFit",
    "LawError",
    "MetricError",
    "Note",
    "PredictionError",
    "ScalingLawModel",
    "Schema",
    "SchemaError",
    "__version__",
    "available_cost_forms",
    "available_laws",
    "build_dataset",
    "compare",
    "cost_model",
    "discover_schema",
    "fit",
    "fit_cost_model",
    "fit_isoflop",
    "isoflop_grid",
    "load",
    "load_config",
    "score_design",
    "suggest_runs",
]


def load(path):
    """Load a fitted model from a ``.slaw`` artifact directory.

    Args:
        path: The artifact directory.

    Returns:
        The reconstructed :class:`~simple_scaling_laws.model.ScalingLawModel`.
    """
    return ScalingLawModel.load(path)
