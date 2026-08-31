"""Column-role discovery and input validation."""

import polars as pl
import pytest

from simple_scaling_laws import fit
from simple_scaling_laws.data import DataError, build_dataset, read_table
from simple_scaling_laws.schema import SchemaError, discover_schema

CONVENTIONAL = [
    "training_run_id",
    "train_set_id",
    "test_set_id",
    "optimizer_seed",
    "model_size__n_params",
    "dataset_size__n_subjects",
    "test_loss__cross_entropy",
    "train_loss__cross_entropy",
    "test_metric__auroc",
    "test_metric__auprc",
    "train_metric__accuracy",
]


def test_conventional_names_need_no_configuration():
    """The documented prefixes are enough to resolve every role."""
    schema = discover_schema(CONVENTIONAL)
    assert schema.training_run_id == "training_run_id"
    assert schema.model_size == ("model_size__n_params",)
    assert schema.dataset_size == ("dataset_size__n_subjects",)
    assert schema.primary_target == "test_loss__cross_entropy"
    assert len(schema.targets) == 5


def test_unrecognized_columns_are_ignored():
    """Bookkeeping columns are harmless."""
    schema = discover_schema([*CONVENTIONAL, "git_sha", "wall_clock_seconds", "notes"])
    assert "git_sha" not in schema.target_names
    assert len(schema.targets) == 5


def test_multiple_predictors_of_one_role_are_all_used():
    """Nothing restricts a design to one model-size or one dataset-size column."""
    schema = discover_schema(
        [
            "training_run_id",
            "model_size__n_params",
            "model_size__n_layers",
            "dataset_size__n_subjects",
            "test_loss__ce",
        ]
    )
    assert schema.model_size == ("model_size__n_params", "model_size__n_layers")
    assert schema.predictors[-1] == "dataset_size__n_subjects"


def test_overrides_handle_non_conventional_names():
    """A table with its own naming can be mapped without renaming columns."""
    schema = discover_schema(
        ["run", "bootstrap_id", "params", "n_train", "loss", "auroc"],
        overrides={
            "training_run_id": "run",
            "test_set_id": "bootstrap_id",
            "model_size": "params",
            "dataset_size": "n_train",
            "test_loss": "loss",
            "test_metric": ["auroc"],
        },
    )
    assert schema.training_run_id == "run"
    assert schema.test_set_id == "bootstrap_id"
    assert schema.primary_target == "loss"
    assert schema.target("auroc").role == "test_metric"


def test_optional_identifiers_may_be_absent():
    """Only the training run identifier is required."""
    schema = discover_schema(["training_run_id", "model_size__n", "dataset_size__d", "test_loss__ce"])
    assert schema.test_set_id is None
    assert schema.train_set_id is None
    assert schema.optimizer_seed is None


def test_missing_run_identifier_is_an_error():
    """Without a run identifier the data model cannot be applied at all."""
    with pytest.raises(SchemaError, match="training_run_id"):
        discover_schema(["model_size__n", "dataset_size__d", "test_loss__ce"])


def test_missing_predictors_is_an_error():
    """A table with no scale columns has no scaling law to fit."""
    with pytest.raises(SchemaError, match="No predictor columns"):
        discover_schema(["training_run_id", "test_loss__ce"])


def test_missing_targets_is_an_error():
    """A table with no measured outcome has nothing to fit."""
    with pytest.raises(SchemaError, match="No target columns"):
        discover_schema(["training_run_id", "model_size__n"])


def test_overrides_must_name_existing_columns():
    """A typo in an override is caught immediately."""
    with pytest.raises(SchemaError, match="not found"):
        discover_schema(["training_run_id", "model_size__n", "test_loss__ce"], {"test_set_id": "nope"})


def test_unknown_override_roles_are_rejected():
    """Only real roles can be overridden."""
    with pytest.raises(SchemaError, match="Unknown column override role"):
        discover_schema(CONVENTIONAL, {"model_shape": "x"})


def test_identifier_overrides_take_exactly_one_column():
    """An identifier is one column, not a list."""
    with pytest.raises(SchemaError, match="exactly one column"):
        discover_schema(CONVENTIONAL, {"training_run_id": ["a", "b"]})


def test_a_column_cannot_be_both_predictor_and_target():
    """Overlapping overrides are rejected rather than silently resolved."""
    with pytest.raises(SchemaError, match="both predictors and targets"):
        discover_schema(
            ["training_run_id", "x", "y"],
            {"model_size": "x", "dataset_size": "y", "test_loss": "x"},
        )


def test_explicit_primary_target_is_honoured():
    """The reference target for correlations can be chosen."""
    schema = discover_schema(CONVENTIONAL, primary_target="test_metric__auroc")
    assert schema.primary_target == "test_metric__auroc"


def test_unknown_primary_target_is_rejected():
    """A primary target must be one of the discovered targets."""
    with pytest.raises(SchemaError, match="not one of the targets"):
        discover_schema(CONVENTIONAL, primary_target="test_metric__nope")


def test_primary_target_falls_back_by_role_priority():
    """Without a test loss, the highest-priority available role supplies the primary target."""
    schema = discover_schema(["training_run_id", "model_size__n", "dataset_size__d", "test_metric__auroc"])
    assert schema.primary_target == "test_metric__auroc"


def test_schema_round_trips_through_a_dictionary():
    """The schema is serialized into the artifact and restored from it."""
    from simple_scaling_laws.schema import Schema

    schema = discover_schema(CONVENTIONAL)
    assert Schema.from_dict(schema.to_dict()) == schema


def test_predictors_must_be_constant_within_a_run():
    """A single trained model cannot have two model sizes."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r1"],
            "model_size__n": [1e6, 1e7],
            "dataset_size__d": [1e4, 1e4],
            "test_loss__ce": [2.0, 1.9],
        }
    )
    with pytest.raises(DataError, match="vary within training run"):
        build_dataset(frame)


def test_predictors_must_be_positive():
    """Predictors are log-scaled, so zero is not a scale."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r2"],
            "model_size__n": [0.0, 1e7],
            "dataset_size__d": [1e4, 1e4],
            "test_loss__ce": [2.0, 1.9],
        }
    )
    with pytest.raises(DataError, match="strictly positive"):
        build_dataset(frame)


def test_empty_tables_are_rejected():
    """An empty table is an error, not an empty fit."""
    frame = pl.DataFrame(
        {"training_run_id": [], "model_size__n": [], "dataset_size__d": [], "test_loss__ce": []}
    )
    with pytest.raises(DataError, match="empty"):
        build_dataset(frame)


def test_parquet_and_csv_are_both_readable(tmp_path, loss_frame):
    """Parquet is the intended format; CSV is accepted for convenience."""
    parquet, csv = tmp_path / "runs.parquet", tmp_path / "runs.csv"
    loss_frame.write_parquet(parquet)
    loss_frame.write_csv(csv)
    assert read_table(parquet).shape == loss_frame.shape
    assert read_table(csv).shape == loss_frame.shape
    assert fit(parquet, n_draws=50, seed=0).manifest["source"]["path"] == str(parquet)


def test_unsupported_file_types_are_rejected(tmp_path):
    """An unknown suffix is refused rather than guessed at."""
    with pytest.raises(DataError, match="Unsupported input suffix"):
        read_table(tmp_path / "runs.json")


def test_a_run_spanning_several_train_sets_is_flagged():
    """One training_run_id should identify one independently trained model."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r1", "r2", "r2"],
            "train_set_id": ["a", "b", "c", "c"],
            "model_size__n": [1e6, 1e6, 1e7, 1e7],
            "dataset_size__d": [1e4, 1e4, 1e4, 1e4],
            "test_loss__ce": [2.0, 2.1, 1.8, 1.9],
        }
    )
    codes = {note.code for note in build_dataset(frame).notes}
    assert "run_spans_train_sets" in codes


def test_duplicate_evaluations_are_flagged():
    """The same model scored twice on the same resample is suspicious enough to report."""
    frame = pl.DataFrame(
        {
            "training_run_id": ["r1", "r1", "r2", "r2"],
            "test_set_id": ["b1", "b1", "b1", "b2"],
            "model_size__n": [1e6, 1e6, 1e7, 1e7],
            "dataset_size__d": [1e4, 1e4, 1e4, 1e4],
            "test_loss__ce": [2.0, 2.1, 1.8, 1.9],
        }
    )
    codes = {note.code for note in build_dataset(frame).notes}
    assert "duplicate_evaluations" in codes


def test_pandas_input_is_accepted(loss_frame):
    """Callers are not forced to adopt Polars."""
    pandas = pytest.importorskip("pandas")
    model = fit(pandas.DataFrame(loss_frame.to_dict(as_series=False)), n_draws=50, seed=0)
    assert model.manifest["n_training_runs"] == 18
