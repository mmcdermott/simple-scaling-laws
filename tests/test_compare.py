"""Comparing fitted systems, which is what a triage platform actually asks for.

The load-bearing property is that differences are computed from *paired* draws. Two arms run at the same
scales share a configuration-level deviation, so the difference between them is pinned down far more sharply
than either level is; differencing unpaired draws throws that away and produces intervals wide enough that no
real difference is ever detectable.
"""

import numpy as np
import polars as pl
import pytest

from simple_scaling_laws import compare, fit
from simple_scaling_laws.compare import ComparisonError
from simple_scaling_laws.metrics import HIGHER, UNIT, MetricKind, from_fitting_scale
from simple_scaling_laws.model import PredictionError
from simple_scaling_laws.simulate import simulate_runs

POINTS = {"model_size__n_params": [1e7, 1e9], "dataset_size__n_subjects": [1e4, 1e4]}


def arm(alpha=0.30, seed=0, offset=0.0, **kwargs):
    """One system's experiment records, differing from the baseline in its exponent or level."""
    return simulate_runs(
        {"test_loss__ce": {"E": 1.0 + offset, "A": 2.0, "alpha": alpha, "B": 1.5, "beta": 0.25}},
        runs_per_config=2,
        evaluations_per_run=4,
        run_sd=0.01,
        eval_sd=0.02,
        seed=seed,
        **kwargs,
    )


@pytest.fixture(scope="module")
def systems():
    """Three arms fit with a common seed, so their draws pair."""
    return {
        "baseline": fit(arm(alpha=0.30, seed=0), n_draws=300, seed=0),
        "better": fit(arm(alpha=0.36, seed=1), n_draws=300, seed=0),
        "middling": fit(arm(alpha=0.33, seed=2), n_draws=300, seed=0),
    }


def test_comparison_is_paired_when_the_fits_share_a_seed_and_a_grid(systems):
    """Pairing is the normal case for arms of one experiment, and is detected automatically."""
    with pytest.warns(UserWarning, match="outside at least one system"):
        table = compare(systems, POINTS, reference="baseline")
    assert table["paired"].to_list() == [True] * 6


def test_every_system_appears_at_every_point(systems):
    """Long format: one row per point per system, so k arms do not become k-squared columns."""
    with pytest.warns(UserWarning):
        table = compare(systems, POINTS)
    assert table.height == 6
    assert set(table["system"].to_list()) == set(systems)
    assert table["target"].unique().to_list() == ["test_loss__ce"]


def test_probabilities_of_being_best_form_a_distribution(systems):
    """Across systems at one point the win probabilities must sum to one."""
    with pytest.warns(UserWarning):
        table = compare(systems, POINTS)
    for _, group in table.group_by("point_id"):
        assert group["p_best"].sum() == pytest.approx(1.0)
    assert (table["p_best"] >= 0).all()


def test_the_steeper_arm_wins_at_larger_scale(systems):
    """A steeper exponent on a loss must win where it matters, and win more the further out."""
    with pytest.warns(UserWarning):
        table = compare(systems, POINTS, reference="baseline")
    far = table.filter(pl.col("point_id") == "1")
    winner = far.sort("p_best", descending=True).row(0, named=True)
    assert winner["system"] == "better"
    assert winner["p_best"] > 0.9
    assert winner["p_better_than_reference"] > 0.9
    # A loss improves downward, so the better arm's difference from the baseline is negative.
    assert winner["difference__median"] < 0


def test_the_difference_interval_is_reported(systems):
    """The size of the win, not just its probability, is what decides whether to scale up."""
    with pytest.warns(UserWarning):
        table = compare(systems, POINTS, reference="baseline", quantiles=[0.025, 0.5, 0.975])
    row = table.filter((pl.col("system") == "better") & (pl.col("point_id") == "1")).row(0, named=True)
    assert row["difference__q025"] <= row["difference__median"] <= row["difference__q975"]
    baseline = table.filter((pl.col("system") == "baseline") & (pl.col("point_id") == "1")).row(0, named=True)
    assert baseline["difference__median"] == pytest.approx(0.0)
    assert baseline["p_better_than_reference"] == pytest.approx(0.0)


def _arms_sharing_configuration_structure(shared_sd, seed=0):
    """Two arms whose residuals share a per-configuration deviation.

    This is what a real pair of arms looks like: both are evaluated on the same splits, and where
    the law misfits at some scale it misfits both curves the same way. That shared part cancels in
    the difference, which is exactly the information pairing preserves and unpaired differencing
    discards.
    """
    rng = np.random.default_rng(seed)
    model_sizes, data_sizes = (1e6, 1e7, 1e8), (1e3, 1e4, 1e5)
    configurations = [(m, d) for m in model_sizes for d in data_sizes]
    common = rng.normal(0.0, shared_sd, len(configurations))
    reference = np.exp(np.log(np.array(configurations)).mean(axis=0))

    frames = []
    for alpha in (0.30, 0.34):
        rows = []
        for c, (model_size, data_size) in enumerate(configurations):
            log_x = np.log(np.array([model_size, data_size])) - np.log(reference)
            value = 1.0 + 2.0 * np.exp(-alpha * log_x[0]) + 1.5 * np.exp(-0.25 * log_x[1])
            for r in range(2):
                offset = rng.normal(0.0, 0.01)
                for e in range(4):
                    rows.append(
                        {
                            "training_run_id": f"c{c}_r{r}",
                            "test_set_id": f"b{e}",
                            "model_size__n_params": model_size,
                            "dataset_size__n_subjects": data_size,
                            "test_loss__ce": value + common[c] + offset + rng.normal(0.0, 0.02),
                        }
                    )
        frames.append(pl.DataFrame(rows))
    return frames


def test_pairing_makes_the_difference_interval_much_tighter():
    """The whole point: unpaired differencing is far too wide when arms share structure.

    Both comparisons use the same two arms and the same data. They differ only in whether the two fits'
    bootstraps used a common multiplier sequence. When the arms share nothing the two agree, which is why the
    shared deviation here is what makes the comparison worth pairing.
    """
    a, b = _arms_sharing_configuration_structure(shared_sd=0.15)
    paired = compare(
        {"a": fit(a, n_draws=400, seed=0), "b": fit(b, n_draws=400, seed=0)},
        POINTS,
        reference="a",
    )
    with pytest.warns(UserWarning, match="could not be paired"):
        unpaired = compare(
            {"a": fit(a, n_draws=400, seed=0), "b": fit(b, n_draws=400, seed=99)},
            POINTS,
            reference="a",
        )

    def width(table):
        row = table.filter((pl.col("system") == "b") & (pl.col("point_id") == "1")).row(0, named=True)
        return row["difference__q975"] - row["difference__q025"]

    assert unpaired["paired"].to_list() == [False] * 4
    # Not marginally tighter: the shared deviation dominates each arm's own uncertainty, so
    # discarding it inflates the difference interval several times over.
    assert width(paired) * 3 < width(unpaired)


def test_an_unpaired_comparison_warns_but_still_answers():
    """A conservative answer beats refusing to answer."""
    with pytest.warns(UserWarning, match="could not be paired"):
        table = compare(
            {"a": fit(arm(seed=0), n_draws=100, seed=0), "b": fit(arm(seed=1), n_draws=100, seed=7)},
            {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]},
            reference="a",
        )
    assert table.height == 2
    assert not table["paired"].any()


def test_a_higher_is_better_metric_flips_the_direction():
    """For AUROC the winner is the arm with the larger value, not the smaller."""

    def auroc_arm(offset, seed):
        logits = simulate_runs(
            {"test_metric__auroc": {"E": 2.2 + offset, "A": -1.0, "alpha": 0.35, "B": -0.6, "beta": 0.3}},
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.01,
            eval_sd=0.02,
            seed=seed,
        )
        return logits.with_columns(
            test_metric__auroc=pl.Series(
                from_fitting_scale(
                    logits["test_metric__auroc"].to_numpy(), MetricKind(UNIT, HIGHER, "registry")
                )
            )
        )

    table = compare(
        {
            "low": fit(auroc_arm(0.0, 0), n_draws=200, seed=0),
            "high": fit(auroc_arm(0.6, 1), n_draws=200, seed=0),
        },
        {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]},
        reference="low",
    )
    assert table["direction"].unique().to_list() == [HIGHER]
    high = table.filter(pl.col("system") == "high").row(0, named=True)
    assert high["p_best"] > 0.9
    assert high["p_better_than_reference"] > 0.9
    assert high["difference__median"] > 0  # higher AUROC is better, so a positive difference wins
    assert 0.0 <= high["value__median"] <= 1.0


def test_new_run_comparison_is_wider_than_mean_comparison(systems):
    """Comparing two single training runs is less certain than comparing their expected curves."""
    with pytest.warns(UserWarning):
        mean = compare(systems, POINTS, reference="baseline", kind="mean")
        new_run = compare(systems, POINTS, reference="baseline", kind="new-run")

    def width(table):
        row = table.filter((pl.col("system") == "better") & (pl.col("point_id") == "1")).row(0, named=True)
        return row["difference__q975"] - row["difference__q025"]

    assert width(new_run) > width(mean)


def test_a_bare_sequence_of_models_is_accepted(systems):
    """Callers who do not care about names should not have to invent them."""
    with pytest.warns(UserWarning):
        table = compare(list(systems.values()), POINTS)
    assert sorted(set(table["system"].to_list())) == ["system_0", "system_1", "system_2"]


def test_comparison_annotates_the_domain(systems):
    """A point outside any system's observed range is extrapolation for the comparison."""
    with pytest.warns(UserWarning):
        table = compare(systems, POINTS)
    assert set(table.filter(pl.col("point_id") == "0")["domain"]) == {"interpolation"}
    assert set(table.filter(pl.col("point_id") == "1")["domain"]) == {"extrapolation"}


def test_fewer_than_two_systems_is_an_error(systems):
    """A comparison needs something to compare against."""
    with pytest.raises(ComparisonError, match="at least two"):
        compare({"only": systems["baseline"]}, POINTS)


def test_systems_fit_with_different_laws_cannot_be_compared():
    """Two different functional forms are not on a common footing."""
    frame = arm()
    separable = fit(frame, law="separable-power", n_draws=50, seed=0)
    multiplicative = fit(frame, law="multiplicative-power", n_draws=50, seed=0)
    with pytest.raises(ComparisonError, match="different laws"):
        compare({"a": separable, "b": multiplicative}, POINTS)


def test_a_target_missing_from_one_system_is_an_error(systems):
    """Comparing on a metric only some arms reported would silently drop arms."""
    with pytest.raises(ComparisonError, match="no fit for target"):
        compare(systems, POINTS, target="test_metric__nonexistent")


def test_an_unknown_reference_is_an_error(systems):
    """A typo in the reference name must not silently produce a reference-free table."""
    with pytest.raises(ComparisonError, match="not one of the systems"):
        compare(systems, POINTS, reference="nope")


def test_an_unknown_kind_is_an_error(systems):
    """The prediction semantics are the same two as everywhere else."""
    with pytest.raises(PredictionError, match="kind must be one of"):
        compare(systems, POINTS, kind="sideways")


def test_indistinguishable_systems_split_the_credit():
    """Two arms fit to identical data are exactly tied, and neither should be handed the win."""
    frame = arm()
    both = fit(frame, n_draws=200, seed=0)
    table = compare(
        {"a": both, "b": both},
        {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]},
        reference="a",
    )
    assert table["p_best"].to_list() == pytest.approx([0.5, 0.5])
    assert np.allclose(table["difference__median"].to_numpy(), 0.0)


def test_an_assumed_direction_is_flagged():
    """A metric the registry does not know gets an assumed direction, said out loud."""
    frames = [
        simulate_runs(
            {"test_metric__bespoke": {"E": 0.9, "A": -0.2, "alpha": 0.3, "B": -0.1, "beta": 0.25}},
            runs_per_config=2,
            evaluations_per_run=4,
            run_sd=0.01,
            eval_sd=0.02,
            seed=seed,
        )
        for seed in (0, 1)
    ]
    models = {name: fit(f, n_draws=100, seed=0) for name, f in zip(("a", "b"), frames, strict=True)}
    with pytest.warns(UserWarning, match="direction is better"):
        table = compare(
            models,
            {"model_size__n_params": [1e7], "dataset_size__n_subjects": [1e4]},
            reference="a",
        )
    assert table["direction"].unique().to_list() == [HIGHER]
