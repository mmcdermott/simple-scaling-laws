"""Cross-metric diagnostics computed at the right level of aggregation.

The question these answer is "do the loss and the metric move together *across trained models*". Answering it
over raw evaluation rows instead counts each trained model once per bootstrap resample and lets shared test-
set noise masquerade as a relationship between the metrics.
"""

import polars as pl

from simple_scaling_laws import fit

LOSS_PARAMS = {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}
AUROC_PARAMS = {"E": 0.92, "A": -0.15, "alpha": 0.4, "B": -0.08, "beta": 0.3}


def _shared_noise_frame() -> pl.DataFrame:
    """Six runs whose metrics are uncorrelated across models but share strong test-set noise.

    Every trained model is scored on the same five bootstrap resamples. A resample that happens to be easy
    lowers the loss and raises the metric for *every* model at once. Across trained models the two are exactly
    uncorrelated by construction; across evaluation rows they look strongly, and spuriously, related.
    """
    loss_by_run = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
    metric_by_run = [0.81, 0.79, 0.80, 0.80, 0.79, 0.81]
    test_effects = [-0.1, -0.05, 0.0, 0.05, 0.1]
    rows = []
    for i, (loss, metric) in enumerate(zip(loss_by_run, metric_by_run, strict=True)):
        for j, effect in enumerate(test_effects):
            rows.append(
                {
                    "training_run_id": f"r{i}",
                    "test_set_id": f"boot_{j}",
                    "model_size__n": 10.0 ** (6 + i),
                    "dataset_size__d": 1e4,
                    "test_loss__ce": loss + 3 * effect,
                    "test_metric__auroc": metric - 2 * effect,
                }
            )
    return pl.DataFrame(rows)


def test_run_level_correlation_is_not_inflated_by_evaluation_rows():
    """Shared evaluation noise must not be mistaken for a relationship between metrics."""
    model = fit(_shared_noise_frame(), n_draws=100, seed=0)
    correlations = model.diagnostics["metric_correlations"]["test_metric__auroc"]
    assert correlations["n_runs"] == 6
    assert correlations["n_evaluation_rows"] == 30
    assert abs(correlations["pearson"]) < 0.05, "run-level correlation should be ~0 by construction"
    assert abs(correlations["evaluation_row_pearson"]) > 0.5, "row-level correlation is inflated"


def test_correlation_table_exposes_both_levels():
    """The reported table shows the run-level estimate next to the misleading row-level one."""
    model = fit(_shared_noise_frame(), n_draws=100, seed=0)
    table = model.metric_correlations()
    assert table["target"].to_list() == ["test_metric__auroc"]
    row = table.row(0, named=True)
    assert row["n_runs"] == 6
    assert row["n_evaluation_rows"] == 30
    assert abs(row["pearson"]) < abs(row["evaluation_row_pearson"])


def test_correlated_targets_are_detected(multi_target_frame):
    """A loss and an AUROC that both improve with scale are strongly negatively correlated."""
    model = fit(multi_target_frame, n_draws=200, seed=0)
    correlations = model.diagnostics["metric_correlations"]
    assert correlations["test_metric__auroc"]["pearson"] < -0.9
    assert correlations["test_metric__auroc"]["spearman"] < -0.9
    assert correlations["train_loss__cross_entropy"]["pearson"] > 0.9


def test_correlation_intervals_are_reported_and_contain_the_estimate(multi_target_frame):
    """Confidence intervals come from resampling configurations, and bracket the point estimate."""
    model = fit(multi_target_frame, n_draws=100, seed=0)
    correlations = model.diagnostics["metric_correlations"]["test_metric__auroc"]
    low, high = correlations["pearson_ci"]
    assert low <= correlations["pearson"] <= high
    assert -1.0 <= low < high <= 1.0


def test_scaling_similarity_compares_fitted_curves(multi_target_frame):
    """Curves are compared through their predictions on a common grid, not their raw parameters."""
    model = fit(multi_target_frame, n_draws=100, seed=0)
    similarity = model.diagnostics["scaling_similarity"]
    assert similarity["test_metric__auroc"]["grid_pearson"] < -0.9
    assert similarity["train_loss__cross_entropy"]["grid_pearson"] > 0.9
    assert similarity["test_metric__auroc"]["n_grid_points"] == 25


def test_primary_target_is_excluded_from_its_own_diagnostics(multi_target_frame):
    """A target is never correlated against itself."""
    model = fit(multi_target_frame, n_draws=50, seed=0)
    assert model.primary_target not in model.diagnostics["metric_correlations"]
    assert model.primary_target not in model.diagnostics["scaling_similarity"]


def test_correlations_use_only_runs_observed_for_both_targets():
    """A target measured on a subset of runs is correlated only over the runs it shares."""
    frame = _shared_noise_frame().with_columns(
        test_metric__auroc=pl.when(pl.col("training_run_id") == "r5")
        .then(None)
        .otherwise(pl.col("test_metric__auroc"))
    )
    model = fit(frame, n_draws=50, seed=0)
    assert model.diagnostics["metric_correlations"]["test_metric__auroc"]["n_runs"] == 5


def test_constant_metric_yields_undefined_correlation():
    """A metric with no variation has no correlation, and that is reported as null, not zero."""
    frame = _shared_noise_frame().with_columns(test_metric__auroc=pl.lit(0.8))
    model = fit(frame, n_draws=50, seed=0)
    correlations = model.diagnostics["metric_correlations"]["test_metric__auroc"]
    assert correlations["pearson"] is None
    assert correlations["pearson_ci"] is None
    assert model.metric_correlations()["pearson"][0] is None
