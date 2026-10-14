"""End-to-end exercise of the ``scaling-laws`` command line."""

import json

import numpy as np
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
    frame = simulate_runs(PARAMS, runs_per_config=2, evaluations_per_run=4, run_sd=0.01, eval_sd=0.02, seed=0)
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


def test_compare_two_artifacts(workspace, capsys):
    """The comparison command answers which arm wins at a scale neither was run at."""
    for name, alpha in (("a", 0.30), ("b", 0.36)):
        frame = simulate_runs(
            {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": alpha, "B": 1.5, "beta": 0.25}},
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.01,
            eval_sd=0.02,
            seed=0 if name == "a" else 1,
        )
        frame.write_parquet(workspace / f"{name}.parquet")
        assert (
            main(
                [
                    "fit",
                    str(workspace / f"{name}.parquet"),
                    "--output",
                    str(workspace / f"{name}.slaw"),
                    "--draws",
                    "200",
                    "--quiet",
                ]
            )
            == 0
        )
    capsys.readouterr()

    output = workspace / "comparison.parquet"
    assert (
        main(
            [
                "compare",
                str(workspace / "points.parquet"),
                "--model",
                f"baseline={workspace / 'a.slaw'}",
                "--model",
                f"tuned={workspace / 'b.slaw'}",
                "--reference",
                "baseline",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    table = pl.read_parquet(output)
    assert table.height == 4
    assert set(table["system"].to_list()) == {"baseline", "tuned"}
    assert table["paired"].all()
    far = table.filter((pl.col("point_id") == "bigger") & (pl.col("system") == "tuned")).row(0, named=True)
    assert far["p_best"] > 0.9
    assert far["p_better_than_reference"] > 0.9


def test_compare_needs_at_least_two_models(workspace, capsys):
    """One artifact is not a comparison."""
    main(
        [
            "fit",
            str(workspace / "runs.parquet"),
            "--output",
            str(workspace / "one.slaw"),
            "--draws",
            "50",
            "--quiet",
        ]
    )
    capsys.readouterr()
    code = main(["compare", str(workspace / "points.parquet"), "--model", f"only={workspace / 'one.slaw'}"])
    assert code == 1
    assert "at least two" in capsys.readouterr().err


def test_an_explicit_option_beats_the_configuration_even_at_its_default_value(workspace):
    """``--law separable-power`` is a choice, not an absence of one, and must not be overridden."""
    config = workspace / "config.yaml"
    config.write_text("law: multiplicative-power\nseed: 7\n")
    output = workspace / "explicit.slaw"
    arguments = ["fit", str(workspace / "runs.parquet"), "--config", str(config), "--output", str(output)]
    assert main([*arguments, "--law", "separable-power", "--seed", "0", "--draws", "50", "--quiet"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["law"]["law"] == "separable-power"
    assert manifest["seed"] == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [(["--draws", "0"], "n_draws must be at least 1"), (["--config", "BAD"], "error:")],
)
def test_malformed_input_is_a_one_line_error_not_a_traceback(workspace, capsys, extra, message):
    (workspace / "bad.yaml").write_text("law: [unclosed\n")
    extra = [str(workspace / "bad.yaml") if part == "BAD" else part for part in extra]
    code = main(["fit", str(workspace / "runs.parquet"), "--output", str(workspace / "x.slaw"), *extra])
    assert code == 1
    assert message in capsys.readouterr().err


def test_allocate_reports_the_frontier_for_each_budget(workspace, capsys):
    artifact = workspace / "fit.slaw"
    assert (
        main(["fit", str(workspace / "runs.parquet"), "-o", str(artifact), "--draws", "60", "--quiet"]) == 0
    )
    output = workspace / "frontier.csv"
    arguments = [
        "allocate",
        str(artifact),
        "--budget",
        "1e12",
        "--budget",
        "1e13",
        "--cost",
        "power:k=6,a=1,b=1",
    ]
    assert main([*arguments, "-o", str(output)]) == 0
    frontier = pl.read_csv(output)
    assert frontier["budget"].to_list() == [1e12, 1e13]
    assert "model_size__n_params__median" in frontier.columns
    assert "cost model: power: k=6, a=1, b=1" in capsys.readouterr().err


def test_allocate_fits_the_parameters_it_was_not_given(workspace, capsys):
    runs = pl.read_parquet(workspace / "runs.parquet").with_columns(
        (6.0 * pl.col("model_size__n_params") * pl.col("dataset_size__n_subjects")).alias("compute__flops")
    )
    runs.write_parquet(workspace / "costed.parquet")
    artifact = workspace / "fit.slaw"
    assert (
        main(["fit", str(workspace / "runs.parquet"), "-o", str(artifact), "--draws", "60", "--quiet"]) == 0
    )
    code = main(
        [
            "allocate",
            str(artifact),
            "--budget",
            "1e12",
            "--cost",
            "power:b=1",
            "--cost-records",
            str(workspace / "costed.parquet"),
        ]
    )
    assert code == 0
    assert "cost model: power: k=6, a=1, b=1" in capsys.readouterr().err


def test_isoflop_profiles_sweeps_and_predicts_beyond_them(tmp_path, capsys):
    from simple_scaling_laws.cost import cost_model

    chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
    design = [
        (float(n), float(chinchilla.dataset_size_for(c, n)))
        for c in (1e12, 1e13, 1e14)
        for n in np.sqrt(c / 60.0) * np.logspace(-0.6, 0.6, 6)
    ]
    runs = simulate_runs(
        {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}},
        configurations=design,
        run_sd=0.005,
        eval_sd=0.005,
    ).with_columns(
        (6.0 * pl.col("model_size__n_params") * pl.col("dataset_size__n_subjects")).alias("compute__flops")
    )
    runs.write_parquet(tmp_path / "sweeps.parquet")
    output = tmp_path / "optimum.csv"
    code = main(
        [
            "isoflop",
            str(tmp_path / "sweeps.parquet"),
            "--cost",
            "power:b=1",
            "--budget",
            "1e16",
            "--draws",
            "100",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    table = pl.read_csv(output)
    assert table["domain"].to_list() == ["extrapolation"]
    assert "dataset_size__n_subjects__median" in table.columns
    err = capsys.readouterr().err
    assert "cost model: power: k=6, a=1, b=1" in err
    assert "optimal model size grows as budget **" in err
