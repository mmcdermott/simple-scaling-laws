"""The ``scaling-laws`` command-line interface.

Two commands do the real work::

    scaling-laws fit runs.parquet --law separable-power --output experiment.slaw
    scaling-laws predict experiment.slaw points.parquet --output predictions.parquet

and a third, ``inspect``, prints a fitted artifact in human-readable form. Statistical options are
deliberately absent from ordinary usage: the package decides them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .api import fit as fit_model
from .artifact import package_version
from .laws import DEFAULT_LAW, available_laws
from .model import DEFAULT_QUANTILES, PREDICTION_KINDS, ScalingLawModel
from .uncertainty import DEFAULT_DRAWS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


class CLIError(ValueError):
    """Raised for malformed command-line input."""


def parse_columns(pairs: Sequence[str] | None) -> dict[str, str] | None:
    """Parse ``--columns role=column`` arguments.

    Args:
        pairs: Raw ``role=column`` strings.

    Returns:
        The override mapping, or ``None`` when nothing was supplied.

    Raises:
        CLIError: If an argument is not of the form ``role=column``.

    Examples:
        >>> parse_columns(["training_run_id=run", "test_set_id=boot"])
        {'training_run_id': 'run', 'test_set_id': 'boot'}
        >>> parse_columns(None) is None
        True
        >>> parse_columns(["oops"])
        Traceback (most recent call last):
        ...
        simple_scaling_laws.cli.CLIError: --columns expects role=column, got 'oops'
    """
    if not pairs:
        return None
    out: dict[str, str] = {}
    for pair in pairs:
        role, sep, column = pair.partition("=")
        if not sep or not role or not column:
            raise CLIError(f"--columns expects role=column, got {pair!r}")
        out[role] = column
    return out


def parse_quantiles(text: str) -> list[float]:
    """Parse a comma-separated list of quantiles.

    Args:
        text: For example ``"0.025,0.5,0.975"``.

    Returns:
        The parsed quantiles.

    Raises:
        CLIError: If a value is not a number in ``[0, 1]``.

    Examples:
        >>> parse_quantiles("0.025,0.5,0.975")
        [0.025, 0.5, 0.975]
        >>> parse_quantiles("0.5,2")
        Traceback (most recent call last):
        ...
        simple_scaling_laws.cli.CLIError: Quantiles must lie in [0, 1], got 2.0
    """
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:
            raise CLIError(f"Cannot parse quantile {piece!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise CLIError(f"Quantiles must lie in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise CLIError("No quantiles supplied")
    return values


def write_frame(frame: pl.DataFrame, path: str | Path | None) -> None:
    """Write a frame to parquet or CSV, or to stdout as CSV when no path is given.

    Args:
        frame: The frame to write.
        path: Destination path, or ``None`` for stdout.

    Raises:
        CLIError: If the path suffix is not ``.parquet`` or ``.csv``.
    """
    if path is None:
        sys.stdout.write(frame.write_csv())
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    match path.suffix.lower():
        case ".parquet" | ".pq":
            frame.write_parquet(path)
        case ".csv":
            frame.write_csv(path)
        case other:
            raise CLIError(f"Unsupported output suffix {other!r}; use .parquet or .csv")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The configured parser.

    Examples:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["fit", "runs.parquet", "--output", "out.slaw"])
        >>> args.command, args.law, args.output
        ('fit', 'separable-power', 'out.slaw')
    """
    parser = argparse.ArgumentParser(
        prog="scaling-laws",
        description="Fit empirical ML scaling laws with conservative uncertainty, and predict at new scales.",
    )
    parser.add_argument("--version", action="version", version=f"simple-scaling-laws {package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit every recognized target in an experiment table.")
    fit_parser.add_argument("input", help="Evaluation records as .parquet or .csv.")
    fit_parser.add_argument(
        "--law",
        default=DEFAULT_LAW,
        choices=sorted(available_laws()),
        help=f"Scaling-law family (default: {DEFAULT_LAW}).",
    )
    fit_parser.add_argument("--output", "-o", required=True, help="Destination .slaw artifact directory.")
    fit_parser.add_argument("--config", help="Optional YAML configuration file.")
    fit_parser.add_argument(
        "--columns",
        action="append",
        metavar="ROLE=COLUMN",
        help="Override a column role, e.g. --columns training_run_id=run. Repeatable.",
    )
    fit_parser.add_argument("--primary-target", help="Target used as the reference for correlations.")
    fit_parser.add_argument(
        "--draws", type=int, default=DEFAULT_DRAWS, help="Bootstrap draws per target (advanced)."
    )
    fit_parser.add_argument("--seed", type=int, default=0, help="Random seed (advanced).")
    fit_parser.add_argument("--quiet", action="store_true", help="Do not print the fit summary.")

    predict_parser = subparsers.add_parser("predict", help="Predict at new scales from a fitted artifact.")
    predict_parser.add_argument("artifact", help="A .slaw artifact directory.")
    predict_parser.add_argument("points", help="Points to predict, as .parquet or .csv.")
    predict_parser.add_argument("--output", "-o", help="Destination .parquet or .csv (default: stdout CSV).")
    predict_parser.add_argument(
        "--target", action="append", help="Restrict to one target. Repeatable; default is all."
    )
    predict_parser.add_argument(
        "--quantiles",
        default=",".join(str(q) for q in DEFAULT_QUANTILES),
        help="Comma-separated quantiles to report.",
    )
    predict_parser.add_argument(
        "--kind",
        default="mean",
        choices=list(PREDICTION_KINDS),
        help="'mean' for the expected scaling curve, 'new-run' to add training stochasticity.",
    )
    predict_parser.add_argument("--seed", type=int, help="Random seed for new-run noise (advanced).")

    inspect_parser = subparsers.add_parser("inspect", help="Print a fitted artifact in readable form.")
    inspect_parser.add_argument("artifact", help="A .slaw artifact directory.")

    return parser


def _run_fit(args: argparse.Namespace) -> int:
    """Execute the ``fit`` subcommand."""
    model = fit_model(
        args.input,
        law=args.law,
        columns=parse_columns(args.columns),
        primary_target=args.primary_target,
        config=args.config,
        n_draws=args.draws,
        seed=args.seed,
    )
    path = model.save(args.output)
    if not args.quiet:
        print(model.summary())
        print(f"\nwrote {path}")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    """Execute the ``predict`` subcommand."""
    model = ScalingLawModel.load(args.artifact)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = model.predict(
            args.points,
            kind=args.kind,
            targets=args.target,
            quantiles=parse_quantiles(args.quantiles),
            seed=args.seed,
        )
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)
    write_frame(predictions, args.output)
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    """Execute the ``inspect`` subcommand."""
    model = ScalingLawModel.load(args.artifact)
    print(model.summary())
    correlations = model.metric_correlations()
    if correlations.height:
        print("\nrun-level correlations with the primary target:")
        print(correlations)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: ``0`` on success, ``1`` on a handled error.
    """
    args = build_parser().parse_args(argv)
    handlers = {"fit": _run_fit, "predict": _run_predict, "inspect": _run_inspect}
    try:
        return handlers[args.command](args)
    except (CLIError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
