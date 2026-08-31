"""Test set-up and fixtures code."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.simulate import simulate_runs

#: A well-behaved separable-power loss used across the test suite.
LOSS_PARAMS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}

#: An AUROC-like metric that increases with scale toward an asymptote from below.
AUROC_PARAMS = {"E": 0.92, "A": -0.15, "alpha": 0.4, "B": -0.08, "beta": 0.3}


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    # You can pass more fixtures here to add them to the namespace
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "np": np,
            "pl": pl,
            "fit": fit,
            "simulate_runs": simulate_runs,
        }
    )


@pytest.fixture
def loss_frame() -> pl.DataFrame:
    """A three-by-three scaling grid with two runs per configuration and moderate noise."""
    return simulate_runs(
        {"test_loss__cross_entropy": LOSS_PARAMS},
        runs_per_config=2,
        evaluations_per_run=5,
        run_sd=0.02,
        eval_sd=0.03,
        seed=0,
    )


@pytest.fixture
def multi_target_frame() -> pl.DataFrame:
    """The same design with a loss, a train loss and an AUROC-like metric."""
    return simulate_runs(
        {
            "test_loss__cross_entropy": LOSS_PARAMS,
            "train_loss__cross_entropy": {"E": 0.7, "A": 2.0, "alpha": 0.35, "B": 1.5, "beta": 0.3},
            "test_metric__auroc": AUROC_PARAMS,
        },
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd={
            "test_loss__cross_entropy": 0.02,
            "train_loss__cross_entropy": 0.02,
            "test_metric__auroc": 0.005,
        },
        eval_sd={
            "test_loss__cross_entropy": 0.03,
            "train_loss__cross_entropy": 0.03,
            "test_metric__auroc": 0.01,
        },
        seed=1,
    )
