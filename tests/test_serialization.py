"""Round-tripping a fitted model through a ``.slaw`` artifact.

The artifact is what the automated experiment platform actually stores and later reloads, so a loaded model
must predict *identically* to the one that was saved -- not approximately.
"""

import json

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import ScalingLawModel, fit, load
from simple_scaling_laws.artifact import (
    DIAGNOSTICS_FILE,
    DRAWS_FILE,
    FITS_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    ArtifactError,
    read_artifact,
)

POINTS = pl.DataFrame(
    {
        "point_id": ["small", "interior", "huge"],
        "model_size__n_params": [1e6, 3e7, 1e11],
        "dataset_size__n_subjects": [1e3, 3e4, 1e6],
    }
)


@pytest.fixture
def saved(tmp_path, multi_target_frame):
    """A fitted model saved to disk, with the original returned alongside."""
    model = fit(multi_target_frame, n_draws=150, seed=0)
    path = model.save(tmp_path / "experiment.slaw")
    return model, path


def test_artifact_contains_exactly_the_documented_files(saved):
    """The artifact is the four documented files, nothing more."""
    _, path = saved
    assert sorted(p.name for p in path.iterdir()) == sorted(
        [MANIFEST_FILE, FITS_FILE, DRAWS_FILE, DIAGNOSTICS_FILE]
    )


def test_loaded_model_predicts_identically(saved):
    """Predictions must agree to the bit, since both use the same stored draws."""
    model, path = saved
    loaded = ScalingLawModel.load(path)
    original = model.predict(POINTS, quantiles=[0.025, 0.1, 0.5, 0.9, 0.975])
    reloaded = loaded.predict(POINTS, quantiles=[0.025, 0.1, 0.5, 0.9, 0.975])
    assert original.columns == reloaded.columns
    assert original.equals(reloaded)


def test_loaded_model_reproduces_new_run_predictions(saved):
    """The random component of a ``new-run`` prediction is seeded from the artifact."""
    model, path = saved
    loaded = load(path)
    assert model.predict(POINTS, kind="new-run").equals(loaded.predict(POINTS, kind="new-run"))


def test_loaded_model_preserves_estimates_and_draws(saved):
    """Parameters, draws, diagnostics and warnings all survive the round trip."""
    model, path = saved
    loaded = ScalingLawModel.load(path)
    assert loaded.targets == model.targets
    assert loaded.primary_target == model.primary_target
    assert loaded.predictors == model.predictors
    for target in model.targets:
        assert loaded.params(target) == pytest.approx(model.params(target))
        assert np.array_equal(loaded.draws[target].params, model.draws[target].params)
        assert np.array_equal(loaded.draws[target].run_sd, model.draws[target].run_sd)
        assert np.array_equal(loaded.draws[target].run_deviations, model.draws[target].run_deviations)
    assert [w.code for w in loaded.warnings] == [w.code for w in model.warnings]
    assert loaded.diagnostics == model.diagnostics


def test_all_targets_get_their_own_draws(saved):
    """Every fitted target is persisted with a full set of parameter draws."""
    model, path = saved
    _, _, draws, _ = read_artifact(path)
    for target in model.targets:
        for name in model.law.param_names:
            assert draws[f"{target}/{name}"].shape == (150,)
        assert draws[f"{target}/run_sd"].shape == (150,)


def test_artifact_json_is_strictly_valid(saved):
    """No bare NaN or Infinity tokens, which are not valid JSON and break strict parsers."""
    _, path = saved
    for name in (MANIFEST_FILE, FITS_FILE, DIAGNOSTICS_FILE):
        text = (path / name).read_text()
        json.loads(text, parse_constant=_reject_constant)


def _reject_constant(token):
    """Fail loudly if a JSON file contains a non-standard constant."""
    raise AssertionError(f"Artifact JSON contains invalid token {token!r}")


def test_manifest_records_provenance_and_design(saved, tmp_path):
    """The manifest describes what was fit, from what, and over what range."""
    model, path = saved
    manifest, _, _, _ = read_artifact(path)
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["law"]["law"] == "separable-power"
    assert manifest["n_configurations"] == 9
    assert manifest["n_training_runs"] == 18
    assert manifest["n_evaluation_rows"] == 18 * 4
    assert manifest["primary_target"] == model.primary_target
    domain = manifest["predictors"]["model_size__n_params"]
    assert domain["min"] == 1e6 and domain["max"] == 1e8
    assert domain["role"] == "model_size"
    assert "created_at" in manifest and "package_version" in manifest


def test_fits_report_both_normalized_and_raw_scale_parameters(saved):
    """Amplitudes are reported in both parameterizations so neither reading is ambiguous."""
    _, path = saved
    _, fits, _, _ = read_artifact(path)
    entry = fits["test_loss__cross_entropy"]
    reference = 1e7  # geometric mean of the model sizes
    expected = entry["params"]["A"] * reference ** entry["params"]["alpha"]
    assert entry["params_raw_scale"]["A"] == pytest.approx(expected)
    assert entry["params_raw_scale"]["alpha"] == pytest.approx(entry["params"]["alpha"])


def test_saving_twice_overwrites_in_place(saved, tmp_path):
    """Re-saving to the same directory replaces it rather than accumulating files."""
    model, path = saved
    model.save(path)
    assert len(list(path.iterdir())) == 4


def test_loading_a_missing_artifact_fails_clearly(tmp_path):
    """A missing directory is an error, not a silent empty model."""
    with pytest.raises(ArtifactError, match="No artifact directory"):
        ScalingLawModel.load(tmp_path / "nope.slaw")


def test_loading_an_incomplete_artifact_fails_clearly(saved):
    """A truncated artifact names the file it is missing."""
    _, path = saved
    (path / FITS_FILE).unlink()
    with pytest.raises(ArtifactError, match=r"fits\.json"):
        ScalingLawModel.load(path)


def test_future_format_versions_are_refused(saved):
    """An artifact from a newer format is refused rather than misread."""
    _, path = saved
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    manifest["format_version"] = "999"
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="format version"):
        ScalingLawModel.load(path)


def test_artifact_is_usable_without_the_original_data(tmp_path, multi_target_frame):
    """Nothing about prediction reaches back to the input table."""
    path = fit(multi_target_frame, n_draws=100, seed=0).save(tmp_path / "e.slaw")
    del multi_target_frame
    loaded = ScalingLawModel.load(path)
    predictions = loaded.predict(POINTS)
    assert predictions.height == 3
    assert not predictions["test_loss__cross_entropy__median"].is_null().any()


def test_saving_into_a_foreign_non_empty_directory_is_refused(tmp_path, saved):
    """A mistyped output path must not scatter files into an unrelated directory."""
    model, _ = saved
    other = tmp_path / "not_an_artifact"
    other.mkdir()
    (other / "important.txt").write_text("do not clobber me")
    with pytest.raises(ArtifactError, match="not an artifact directory"):
        model.save(other)
    assert (other / "important.txt").read_text() == "do not clobber me"


def test_saving_into_an_empty_directory_is_allowed(tmp_path, saved):
    """An empty directory is a perfectly ordinary destination."""
    model, _ = saved
    empty = tmp_path / "fresh.slaw"
    empty.mkdir()
    assert model.save(empty).is_dir()
