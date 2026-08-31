"""End-to-end exercise of the ``scaling-laws`` command line."""

import json

import polars as pl
import pytest

from simple_scaling_laws.cli import CLIError, main, parse_columns, parse_quantiles
from simple_scaling_laws.simulate import simulate_runs

PARAMS = {
    "test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25},
    "test_metric__auroc": {"E": 0.92, "A": -0.15, "alpha": 0.4, "B": -0.08, "beta": 0.3},
}


@pytest.fixture
def workspace(tmp_path):
    """A directory containing an input table and a set of prediction points."""
    frame = simulate_runs(
        PARAMS, runs_per_config=2, evaluations_per_run=4, run_sd=0.01, eval_sd=0.02, seed=0
    )
    frame.write_parquet(tmp_path / "runs.parquet")
    pl.DataFrame(
        {
            "point_id": ["interior", "bigger"],
            "model_size__n_params": [1e7, 1e9],
            "dataset_size__n_subjects": [1e4, 1e5],
        }
    ).write_parquet(tmp_path / "points.parquet")
    return tmp_path


def test_fit_then_predict(workspace, capsys):
    """The documented two-command workflow produces an artifact and a prediction table."""
    artifact = workspace / "experiment.slaw"
    assert main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "100"]) == 0
    summary = capsys.readouterr().out
    assert "separable-power" in summary
    assert "test_loss__cross_entropy (primary)" in summary
    assert artifact.is_dir()

    output = workspace / "predictions.parquet"
    assert (
        main(
            [
                "predict",
                str(artifact),
                str(workspace / "points.parquet"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    predictions = pl.read_parquet(output)
    assert predictions["point_id"].to_list() == ["interior", "bigger"]
    assert predictions["domain"].to_list() == ["interpolation", "extrapolation"]
    for target in PARAMS:
        assert f"{target}__median" in predictions.columns
    assert "outside the observed scaling domain" in capsys.readouterr().err


def test_predict_writes_csv_to_stdout_by_default(workspace, capsys):
    """Leaving out --output makes the command pipeable."""
    artifact = workspace / "experiment.slaw"
    main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "50", "--quiet"])
    capsys.readouterr()
    assert main(["predict", str(artifact), str(workspace / "points.parquet")]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("point_id,")
    assert len(out.splitlines()) == 3


def test_predict_target_and_quantile_selection(workspace, capsys):
    """Targets and quantiles can be narrowed for a compact answer."""
    artifact = workspace / "experiment.slaw"
    main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "50", "--quiet"])
    output = workspace / "one.csv"
    main(
        [
            "predict",
            str(artifact),
            str(workspace / "points.parquet"),
            "--target",
            "test_metric__auroc",
            "--quantiles",
            "0.1,0.5,0.9",
            "--output",
            str(output),
        ]
    )
    predictions = pl.read_csv(output)
    assert "test_metric__auroc__median" in predictions.columns
    assert "test_metric__auroc__q100" in predictions.columns
    assert "test_metric__auroc__q900" in predictions.columns
    assert not any(c.startswith("test_loss") for c in predictions.columns)


def test_predict_new_run_kind(workspace):
    """``--kind new-run`` widens the intervals with training stochasticity."""
    artifact = workspace / "experiment.slaw"
    main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "200", "--quiet"])
    for kind in ("mean", "new-run"):
        main(
            [
                "predict",
                str(artifact),
                str(workspace / "points.parquet"),
                "--kind",
                kind,
                "--output",
                str(workspace / f"{kind}.parquet"),
            ]
        )
    column = "test_loss__cross_entropy"
    mean = pl.read_parquet(workspace / "mean.parquet")
    new_run = pl.read_parquet(workspace / "new-run.parquet")
    mean_width = mean[f"{column}__q975"] - mean[f"{column}__q025"]
    new_width = new_run[f"{column}__q975"] - new_run[f"{column}__q025"]
    assert (new_width > mean_width).all()


def test_fit_accepts_a_configuration_file(workspace, capsys):
    """A YAML config supplies the law and any column overrides."""
    frame = pl.read_parquet(workspace / "runs.parquet").rename({"training_run_id": "run"})
    frame.write_parquet(workspace / "renamed.parquet")
    config = workspace / "config.yaml"
    config.write_text("law: multiplicative-power\ncolumns:\n  training_run_id: run\n")
    assert (
        main(
            [
                "fit",
                str(workspace / "renamed.parquet"),
                "--config",
                str(config),
                "--output",
                str(workspace / "configured.slaw"),
                "--draws",
                "50",
            ]
        )
        == 0
    )
    manifest = json.loads((workspace / "configured.slaw" / "manifest.json").read_text())
    assert manifest["law"]["law"] == "multiplicative-power"
    assert manifest["schema"]["training_run_id"] == "run"


def test_fit_accepts_inline_column_overrides(workspace):
    """``--columns role=column`` avoids needing a config file for a small remapping."""
    pl.read_parquet(workspace / "runs.parquet").rename({"training_run_id": "run"}).write_parquet(
        workspace / "renamed.parquet"
    )
    assert (
        main(
            [
                "fit",
                str(workspace / "renamed.parquet"),
                "--columns",
                "training_run_id=run",
                "--output",
                str(workspace / "remapped.slaw"),
                "--draws",
                "50",
                "--quiet",
            ]
        )
        == 0
    )


def test_inspect_prints_a_readable_summary(workspace, capsys):
    """``inspect`` describes a stored artifact without re-reading the input table."""
    artifact = workspace / "experiment.slaw"
    main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "50", "--quiet"])
    capsys.readouterr()
    assert main(["inspect", str(artifact)]) == 0
    out = capsys.readouterr().out
    assert "law: separable-power" in out
    assert "run-level correlations" in out
    assert "test_metric__auroc" in out


def test_missing_input_returns_an_error_code(workspace, capsys):
    """A missing input file is a clean exit-1, not a traceback."""
    assert main(["fit", str(workspace / "nope.parquet"), "--output", str(workspace / "x.slaw")]) == 1
    assert "error:" in capsys.readouterr().err


def test_unknown_law_is_rejected_by_the_parser(workspace):
    """The law choice is validated by argparse itself."""
    with pytest.raises(SystemExit):
        main(["fit", str(workspace / "runs.parquet"), "--law", "quadratic", "--output", "x.slaw"])


def test_bad_output_suffix_returns_an_error_code(workspace, capsys):
    """Prediction output must be parquet or CSV."""
    artifact = workspace / "experiment.slaw"
    main(["fit", str(workspace / "runs.parquet"), "--output", str(artifact), "--draws", "50", "--quiet"])
    code = main(
        [
            "predict",
            str(artifact),
            str(workspace / "points.parquet"),
            "--output",
            str(workspace / "out.xlsx"),
        ]
    )
    assert code == 1
    assert "Unsupported output suffix" in capsys.readouterr().err


def test_parse_columns_rejects_malformed_pairs():
    """``--columns`` needs ``role=column``."""
    assert parse_columns(["a=b"]) == {"a": "b"}
    with pytest.raises(CLIError):
        parse_columns(["="])


def test_parse_quantiles_rejects_out_of_range_values():
    """Quantiles outside [0, 1] are rejected before any computation happens."""
    assert parse_quantiles(" 0.1 , 0.9 ") == [0.1, 0.9]
    with pytest.raises(CLIError):
        parse_quantiles("0.5,-1")
    with pytest.raises(CLIError):
        parse_quantiles("abc")
    with pytest.raises(CLIError):
        parse_quantiles(",")
