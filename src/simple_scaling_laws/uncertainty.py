"""Nonparametric uncertainty for a fitted scaling law.

The default is a **wild cluster bootstrap over scaling configurations**, computed on run-level
residuals. It was chosen because it is the only cheap scheme that is simultaneously:

* *assumption-free* about the distribution of the errors -- it reuses the observed residuals rather
  than sampling from an assumed Gaussian;
* *valid with very few clusters*, which is the regime this package targets (four to a dozen
  configurations). Webb's six-point weights are used rather than plain sign flips precisely because
  a handful of clusters admits only ``2**n_configurations`` distinct sign patterns;
* *conservative*, because residuals around the fitted curve contain the law's own misspecification
  at each observed scale, so that error is inherited by the intervals instead of being assumed away;
* *design-preserving*: the ``(N, D)`` grid is chosen by the experimenter, not sampled, so resampling
  configurations with replacement would be both wrong in principle and frequently degenerate
  (dropping configurations makes the law unidentifiable). Sign-flipping keeps the design fixed.

Signs are drawn once per **configuration** and shared by every run at that configuration, so the
part of a residual that is common to a scale -- the local lack of fit -- is preserved rather than
averaged away.

Each configuration's pseudo-residual is **floored at the noise the fit absorbed there**. A law with
nearly as many parameters as configurations passes almost exactly through some of them, so their
residuals are close to zero whatever the noise was, and resampling them perturbs nothing -- least of
all the points a nonlinear parameter such as a break location rests on. The run-to-run and evaluation
variances are estimated separately, so the share a configuration's leverage absorbed is known, and the
floor puts it back. It never shrinks a configuration's pseudo-residual, so it pushes intervals wider,
not narrower; see :func:`corrected_residuals` for what it measured.

The wild cluster bootstrap is from Cameron, Gelbach & Miller (2008), *Review of Economics and Statistics*
90:414-427; the six-point weights are Webb's (2023), *Canadian Journal of Economics* 56:839-858; the HC2
leverage correction is MacKinnon & White's (1985), *Journal of Econometrics* 29:305-325. The cluster
degrees-of-freedom and tail corrections and the noise floor below are package-specific, calibrated by
simulation.

When the fit has no residual degrees of freedom left, no wild bootstrap is possible; the package
falls back to a parametric bootstrap from the estimated variance components and says so in a note.

The refits are deliberately *not* parallelized. Two cheap changes -- caching the variable-projection
restarts across draws, and dropping SciPy's Jacobian-based parameter rescaling, which the normalized
predictors make redundant -- brought a thousand draws down to a couple of seconds per target. Process
parallelism would add pickling, a worker pool and a source of nondeterminism to save a handful of
seconds, which is a bad trade for a package whose other goal is being small enough to audit.
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from .fitting import CurveFit, GridSeeder, fit_curve, leverage
from .notes import Note

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fitting import Bounds
    from .laws import LawInstance

#: Default number of bootstrap draws retained per target.
DEFAULT_DRAWS = 1000

#: Webb's six-point weights: mean zero, variance one, and six distinct values so that a handful of
#: clusters still generates a rich set of distinct bootstrap datasets.
WEBB_WEIGHTS: tuple[float, ...] = (
    -((3 / 2) ** 0.5),
    -1.0,
    -((1 / 2) ** 0.5),
    (1 / 2) ** 0.5,
    1.0,
    (3 / 2) ** 0.5,
)

#: Leverage is clipped below one before the residual correction, so a saturated observation cannot
#: produce an infinite pseudo-residual. The cap is tied to the sample size rather than fixed at some
#: constant like 0.99, because a fixed constant would silently let an arbitrary number set the width
#: of the whole interval whenever one configuration is nearly saturated -- which is common when the
#: design barely supports the law.
MAX_LEVERAGE_HEADROOM = 2.0

#: A refit whose cost exceeds the best grid-restart cost by more than this relative margin is
#: re-polished from the grid, so warm-starting cannot silently trap draws in the original basin.
WARM_START_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class Uncertainty:
    """Bootstrap draws for one target.

    Attributes:
        params: Draws of the parameter vector, shape ``(n_draws, n_params)``.
        run_sd: Draws of the run-to-run standard deviation, shape ``(n_draws,)``.
        run_deviations: Empirical, centered run-level deviations at replicated configurations.
            Used to resample training-run noise without assuming it is Gaussian.
        method: ``"wild-cluster"`` or ``"parametric"``.
        n_failed: Number of draws whose refit did not converge (they are retained regardless).
        notes: Warnings raised while estimating uncertainty.
    """

    params: np.ndarray
    run_sd: np.ndarray
    run_deviations: np.ndarray
    method: str
    n_failed: int
    notes: tuple[Note, ...] = ()

    @property
    def n_draws(self) -> int:
        """Number of retained draws."""
        return int(self.params.shape[0])

    def quantiles(self, q: float | np.ndarray) -> np.ndarray:
        """Quantiles of each parameter across draws."""
        return np.quantile(self.params, q, axis=0)


def target_seed(seed: int, target: str) -> int:
    """The random seed used for anything random about one target: its bootstrap, its predictions.

    Derived from the target's *name* rather than its position, so that the same target draws the
    same cluster multipliers in two different experiments even when those experiments report
    different sets of metrics. That is what lets :func:`simple_scaling_laws.compare` difference two
    fits' draws and get the variance of the difference right rather than the sum of two variances.
    It is also why a target's ``new-run`` predictions and correlation interval do not change when
    other targets are added to or dropped from a request.

    Args:
        seed: The experiment-level seed.
        target: The target column name.

    Returns:
        A seed in ``[0, 2**32)``.

    Examples:
        >>> target_seed(0, "test_loss__ce") == target_seed(0, "test_loss__ce")
        True
        >>> target_seed(0, "test_loss__ce") == target_seed(0, "test_metric__auroc")
        False
    """
    return (seed + zlib.crc32(target.encode())) % (2**32)


def _webb_signs(rng: np.random.Generator, n_clusters: int, n_draws: int) -> np.ndarray:
    """Draw Webb six-point multipliers, one per cluster per bootstrap draw."""
    return rng.choice(np.asarray(WEBB_WEIGHTS), size=(n_draws, n_clusters))


def cluster_correction(n_clusters: int, n_params: int) -> float:
    """Small-sample inflation for a bootstrap clustered over very few units.

    The wild bootstrap resamples one multiplier per cluster, so the effective sample size behind an
    interval is the number of *configurations* -- not the number of runs, and certainly not the
    number of evaluation rows. Two things follow, and both were measured rather than assumed.

    First, fitting consumes ``n_params`` of those degrees of freedom, and the per-observation
    leverage correction does not account for that at the cluster level. Simulating from a known law
    and counting how often the nominal 95% intervals actually covered it gave 74% at nine
    configurations and 87% at sixteen; the ratio of the true sampling spread to the bootstrap's was
    1.50 and 1.18, which is ``sqrt(C / (C - p))`` to two decimals in both cases.

    Second, a percentile interval from the bootstrap distribution behaves like a normal interval,
    while the right reference at these sample sizes is Student's t on ``C - 1`` degrees of freedom.
    That accounts for the few points of undercoverage the first factor leaves behind.

    Together they take measured coverage from 74-89% to roughly nominal across the designs this
    package targets. The factor is deliberately not capped: a design that barely supports the law
    should produce intervals that say so.

    Args:
        n_clusters: Number of resampling clusters, i.e. distinct scaling configurations.
        n_params: Number of fitted parameters.

    Returns:
        A factor of at least one.

    Examples:
        >>> round(cluster_correction(9, 5), 3)
        1.765
        >>> round(cluster_correction(16, 5), 3)
        1.312

        The correction grows sharply as the design approaches the point where it can no longer
        support the law, which is the honest direction:

        >>> round(cluster_correction(6, 5), 3)
        3.213
    """
    degrees_of_freedom = float(np.sqrt(n_clusters / max(n_clusters - n_params, 1)))
    return degrees_of_freedom * tail_correction(n_clusters)


def tail_correction(n_clusters: int) -> float:
    """Ratio of Student's t on ``n_clusters - 1`` degrees of freedom to the normal, at 97.5%.

    The second factor of :func:`cluster_correction`, also used on its own to scale the noise floor in
    :func:`corrected_residuals`.

    Args:
        n_clusters: Number of resampling clusters.

    Returns:
        A factor of at least one, approaching one as clusters accumulate.

    Examples:
        >>> round(tail_correction(9), 3)
        1.177
        >>> round(tail_correction(1000), 3)
        1.001
    """
    return float(stats.t.ppf(0.975, max(n_clusters - 1, 1)) / stats.norm.ppf(0.975))


def corrected_residuals(
    law: LawInstance,
    params: np.ndarray,
    log_x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray | None = None,
    noise_var: np.ndarray | None = None,
) -> np.ndarray:
    """Residuals inflated to undo the shrinkage caused by fitting.

    Least squares pulls the curve toward each observation, so raw residuals are systematically
    smaller than the errors they estimate; dividing by ``sqrt(1 - h_r)`` restores the
    per-observation scale (the HC2 correction). Then :func:`cluster_correction` accounts for the
    degrees of freedom the fit consumed at the level the bootstrap actually resamples at. Resampling
    uncorrected residuals gives intervals that are demonstrably far too narrow at the sample sizes
    this package targets.

    Both are *average* corrections, and they fail where leverage is uneven. A configuration with
    cluster leverage ``H_c`` near one keeps only ``1 - H_c`` of its noise in its residual, and no
    rescaling of a residual near zero recovers it. So, when the noise variance is supplied, each
    configuration's mean pseudo-residual is also floored at ``tail * sqrt(H_c * v_c)``, where
    ``v_c`` is the variance of that configuration's weighted mean and ``tail`` is
    :func:`tail_correction`: the share of its noise the fit absorbed, which its residual cannot
    show. Within-configuration deviations are left as they are, so the floor acts on each
    configuration's mean -- the part a refit responds to -- and an individual run's pseudo-residual
    can still move toward zero when its deviation points the other way.

    The floor is measured, not assumed. On the ``broken-power`` law -- six parameters on eight or
    twelve scales -- nominal 95% prediction intervals covered the truth 70-88% of the time without
    it and 90-97% with it (60 replications per design, three points inside and beyond the sweep).
    On the other laws' recommended designs, which already covered 82-98%, it moved coverage to
    93-98% and widened median intervals by at most about 15%. Under a misspecified law it changed
    nothing measurable, because there the residuals are already larger than the floor. A purely
    additive correction, ``m_c**2 + H_c * v_c`` in place of both average factors, calibrated as well
    with narrower intervals when the law was right, but lost half its coverage of the true curve
    under misspecification, which is why the floor is a maximum rather than a replacement.

    Args:
        law: The instantiated law.
        params: The fitted parameter vector.
        log_x: Log-normalized predictors, one row per run.
        y: Run-level means.
        weights: Fitting weights.
        clusters: Resampling cluster (configuration) of each run. Defaults to one per observation.
        noise_var: Variance of each run mean, run-to-run plus evaluation. Without it, no floor is
            applied.

    Returns:
        Corrected residuals, one per run.

    Examples:
        A sweep that the law passes through exactly has nothing left to resample, unless the noise
        it absorbed is put back:

        >>> from .laws import build_law
        >>> law = build_law("separable-power", ("model_size__n",), ())
        >>> params = np.array([1.0, 2.0, 0.3])
        >>> log_x = np.log(np.array([[0.1], [1.0], [10.0], [100.0]]))
        >>> exact = law.evaluate(params, log_x)
        >>> corrected_residuals(law, params, log_x, exact, np.ones(4))
        array([0., 0., 0., 0.])
        >>> noise_var = np.full(4, 1e-4)
        >>> floored = corrected_residuals(law, params, log_x, exact, np.ones(4), np.arange(4), noise_var)
        >>> bool(np.all(floored > 0))
        True
    """
    residual = y - law.evaluate(params, log_x)
    absorbed = leverage(law, params, log_x, weights)
    return pseudo_residuals(residual, absorbed, weights, law.n_params, clusters, noise_var)


def pseudo_residuals(
    residual: np.ndarray,
    absorbed: np.ndarray,
    weights: np.ndarray,
    n_params: int,
    clusters: np.ndarray | None = None,
    noise_var: np.ndarray | None = None,
) -> np.ndarray:
    """The corrections of :func:`corrected_residuals`, for any fit whose residuals and leverages are known.

    Shared with estimators that are not scaling laws -- the isoFLOP profile fit -- so that every
    bootstrap in the package corrects its residuals the same, calibrated way.

    Args:
        residual: Raw residuals, one per run.
        absorbed: Each run's leverage.
        weights: Fitting weights.
        n_params: Parameters the fit consumed.
        clusters: Resampling cluster of each run. Defaults to one per observation.
        noise_var: Variance of each run mean; without it, no noise floor is applied.

    Returns:
        Corrected residuals, one per run.
    """
    ceiling = 1.0 - MAX_LEVERAGE_HEADROOM / max(residual.size, MAX_LEVERAGE_HEADROOM + 1.0)
    cluster = np.arange(residual.size) if clusters is None else np.unique(clusters, return_inverse=True)[1]
    n_clusters = int(cluster.max()) + 1
    corrected = residual / np.sqrt(1.0 - np.clip(absorbed, 0.0, ceiling))
    corrected *= cluster_correction(n_clusters, n_params)
    if noise_var is None:
        return corrected

    total = np.bincount(cluster, weights)
    mean = np.bincount(cluster, weights * corrected) / total
    mean_var = np.bincount(cluster, weights**2 * noise_var) / total**2
    floor = np.sqrt(np.minimum(np.bincount(cluster, absorbed), 1.0) * mean_var) * tail_correction(n_clusters)
    floored = np.copysign(np.maximum(np.abs(mean), floor), mean)
    return corrected + (floored - mean)[cluster]


def _refit(
    law: LawInstance,
    log_x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bounds: Bounds,
    params: np.ndarray,
    seeder: GridSeeder,
) -> CurveFit:
    """Refit one bootstrap dataset, warm-started but guarded against a bad basin.

    Warm-starting every draw from the point estimate is what makes the bootstrap affordable, but on
    its own it could trap every draw in the point estimate's basin and understate uncertainty. The
    guard is cheap: the variable-projection grid is re-scored on the resampled data (a handful of
    matrix products against pre-factorized designs), and if any grid point already beats the
    warm-started optimum the draw is re-polished from there instead.
    """
    warm = fit_curve(law, log_x, y, weights, bounds, starts=[params])
    cost, seed = seeder.seeds(y, n_seeds=1, exact=False)[0]
    if cost < warm.cost * (1.0 - WARM_START_TOLERANCE):
        cold = fit_curve(law, log_x, y, weights, bounds, starts=[seed])
        if cold.cost < warm.cost:
            return cold
    return warm


def _run_deviations(mean: np.ndarray, config_index: np.ndarray) -> np.ndarray:
    """Centered run-level deviations at configurations with replicate runs.

    Deviations are rescaled by ``sqrt(n_c / (n_c - 1))`` so their variance estimates the run-level
    variance rather than the (smaller) variance of a deviation from an estimated configuration mean.
    """
    deviations: list[np.ndarray] = []
    for config in np.unique(config_index):
        members = np.flatnonzero(config_index == config)
        if members.size < 2:
            continue
        values = mean[members]
        deviations.append((values - values.mean()) * np.sqrt(members.size / (members.size - 1)))
    return np.concatenate(deviations) if deviations else np.zeros(0)


def _config_variance_parts(
    mean: np.ndarray, config_index: np.ndarray, per_run_var: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-configuration sums of squares, evaluation-noise corrections and degrees of freedom."""
    sums, corrections, dofs = [], [], []
    for config in np.unique(config_index):
        members = np.flatnonzero(config_index == config)
        n_c = members.size
        if n_c < 2:
            continue
        values = mean[members]
        sums.append(float(((values - values.mean()) ** 2).sum()))
        corrections.append((1.0 - 1.0 / n_c) * float(per_run_var[members].sum()))
        dofs.append(n_c - 1)
    return np.array(sums), np.array(corrections), np.array(dofs, dtype=float)


def _run_sd_draws(
    rng: np.random.Generator,
    mean: np.ndarray,
    config_index: np.ndarray,
    per_run_var: np.ndarray,
    n_draws: int,
    fallback: float,
    from_replicates: bool = True,
) -> np.ndarray:
    """Draws of the run-level standard deviation.

    When the point estimate came from replicate runs, configurations are resampled with replacement and
    their sum-of-squares contributions recombined -- a nonparametric bootstrap of the pooled estimate that
    assumes nothing about the shape of the run-to-run distribution. Otherwise the point estimate is
    repeated. That includes the case where a few replicates exist but too few to be trusted, and the fit
    used the residual scatter instead: resampling those replicates anyway would describe a different
    estimate from the one reported, and one near-duplicate pair would put every draw at zero.
    """
    sums, corrections, dofs = _config_variance_parts(mean, config_index, per_run_var)
    if sums.size == 0 or not from_replicates:
        return np.full(n_draws, fallback, dtype=float)
    index = rng.integers(0, sums.size, size=(n_draws, sums.size))
    total = sums[index].sum(axis=1) - corrections[index].sum(axis=1)
    return np.sqrt(np.maximum(0.0, total / dofs[index].sum(axis=1)))


def bootstrap(
    law: LawInstance,
    log_x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bounds: Bounds,
    params: np.ndarray,
    config_index: np.ndarray,
    per_run_var: np.ndarray,
    run_sd: float,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    target: str = "target",
    run_sd_from_replicates: bool = True,
    n_design_configurations: int | None = None,
) -> Uncertainty:
    """Estimate parameter uncertainty by wild cluster bootstrap over configurations.

    Args:
        law: The instantiated law.
        log_x: Log-normalized predictors, one row per run.
        y: Run-level means.
        weights: Fitting weights.
        bounds: Parameter box constraints.
        params: The fitted parameter vector.
        config_index: Configuration index of each run; the bootstrap clusters on this.
        per_run_var: Evaluation-noise variance of each run mean.
        run_sd: Point estimate of the run-level standard deviation.
        n_draws: Number of draws to retain.
        seed: Seed for the random number generator.
        target: Name of the target, used only in warning messages.
        run_sd_from_replicates: Whether ``run_sd`` was estimated from replicate runs, and so may be
            resampled from them; otherwise its draws repeat it.
        n_design_configurations: Configurations in the whole experiment. When this target was
            observed at all of them, a shortage of clusters is a property of the design and is
            already reported as ``too_few_configurations``, so it is not repeated here.

    Returns:
        The :class:`Uncertainty` for this target.

    Examples:
        >>> from .fitting import fit_curve, parameter_bounds
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> grid = np.array([[n, d] for n in (1.0, 10.0, 100.0) for d in (1.0, 10.0, 100.0)])
        >>> log_x = np.log(grid)
        >>> rng = np.random.default_rng(0)
        >>> y = law.evaluate(np.array([0.5, 1.0, 2.0, 0.3, 0.6]), log_x) + rng.normal(0, 0.02, 9)
        >>> weights, bounds = np.ones(9), parameter_bounds(law, y, signed_amplitude=False)
        >>> fit = fit_curve(law, log_x, y, weights, bounds)
        >>> draws = bootstrap(
        ...     law,
        ...     log_x,
        ...     y,
        ...     weights,
        ...     bounds,
        ...     fit.params,
        ...     np.arange(9),
        ...     np.zeros(9),
        ...     run_sd=0.0,
        ...     n_draws=64,
        ...     seed=0,
        ... )
        >>> draws.method, draws.params.shape
        ('wild-cluster', (64, 5))

        The draws bracket the fitted point estimate:

        >>> lo, hi = draws.quantiles([0.025, 0.975])
        >>> bool(np.all(lo <= fit.params) and np.all(fit.params <= hi))
        True
    """
    rng = np.random.default_rng(seed)
    notes: list[Note] = []
    fitted = law.evaluate(params, log_x)
    deviations = _run_deviations(y, config_index)

    clusters, cluster_of_run = np.unique(config_index, return_inverse=True)
    residual_dof = y.size - law.n_params
    pseudo = (
        corrected_residuals(law, params, log_x, y, weights, config_index, run_sd**2 + per_run_var)
        if residual_dof > 0
        else np.zeros(0)
    )
    method = "wild-cluster"
    if residual_dof <= 0 or not np.any(np.abs(pseudo) > 0):
        method = "parametric"
        notes.append(
            Note(
                "parametric_uncertainty",
                "warning",
                "The fit has no usable residual variation left (there are as many parameters as "
                "distinct observations, or the fit is exact), so uncertainty had to be simulated "
                "from the estimated variance components instead of resampled from residuals. "
                "Intervals rest on a normality assumption and understate model misspecification.",
                {"residual_dof": int(residual_dof), "n_runs": int(y.size), "n_params": law.n_params},
            )
        )

    if method == "wild-cluster":
        multipliers = _webb_signs(rng, clusters.size, n_draws)[:, cluster_of_run]
        synthetic = fitted[None, :] + multipliers * pseudo[None, :]
        missing_here = n_design_configurations is None or clusters.size < n_design_configurations
        if clusters.size < law.n_params + 2 and missing_here:
            notes.append(
                Note(
                    "few_bootstrap_clusters",
                    "warning",
                    f"Uncertainty for {target!r} is resampled over only {clusters.size} scaling "
                    f"configuration(s) for {law.n_params} parameters. Intervals are unreliable and "
                    "probably too narrow; add configurations rather than more runs or evaluations.",
                    {
                        "target": target,
                        "n_configurations": int(clusters.size),
                        "n_params": law.n_params,
                    },
                )
            )
    else:
        variance = run_sd**2 + per_run_var
        if not np.any(variance > 0):
            notes.append(
                Note(
                    "no_uncertainty_information",
                    "error",
                    f"There is no information at all from which to estimate uncertainty for "
                    f"{target!r}: the fit has no residual degrees of freedom, no configuration has "
                    "replicate training runs, and no run has repeated evaluations. The reported "
                    "intervals have zero width and mean nothing -- do not read them as confidence.",
                    {"target": target, "n_runs": int(y.size), "n_params": law.n_params},
                )
            )
        sd = np.sqrt(np.maximum(variance, 0.0))
        synthetic = fitted[None, :] + rng.normal(0.0, 1.0, size=(n_draws, y.size)) * sd[None, :]

    seeder = GridSeeder(law, log_x, weights, bounds)
    draws = np.empty((n_draws, law.n_params), dtype=float)
    n_failed = 0
    for b in range(n_draws):
        refit = _refit(law, log_x, synthetic[b], weights, bounds, params, seeder)
        draws[b] = refit.params
        n_failed += not refit.success
    if n_failed:
        notes.append(
            Note(
                "bootstrap_nonconvergence",
                "info",
                f"{n_failed} of {n_draws} bootstrap refits did not report convergence; they are "
                "retained, which widens rather than narrows the intervals.",
                {"n_failed": int(n_failed), "n_draws": int(n_draws)},
            )
        )
    return Uncertainty(
        params=draws,
        run_sd=_run_sd_draws(rng, y, config_index, per_run_var, n_draws, run_sd, run_sd_from_replicates),
        run_deviations=deviations,
        method=method,
        n_failed=n_failed,
        notes=tuple(notes),
    )


def to_arrays(law: LawInstance, uncertainty: Uncertainty) -> dict[str, np.ndarray]:
    """Flatten draws into a name-keyed mapping for serialization.

    Args:
        law: The instantiated law, supplying parameter names.
        uncertainty: The draws to flatten.

    Returns:
        A mapping from parameter name (plus ``run_sd`` and ``run_deviations``) to arrays.
    """
    arrays: dict[str, np.ndarray] = {name: uncertainty.params[:, i] for i, name in enumerate(law.param_names)}
    arrays["run_sd"] = uncertainty.run_sd
    arrays["run_deviations"] = uncertainty.run_deviations
    return arrays


def from_arrays(law: LawInstance, arrays: dict[str, Any], method: str, n_failed: int) -> Uncertainty:
    """Rebuild an :class:`Uncertainty` from :func:`to_arrays` output."""
    return Uncertainty(
        params=np.column_stack([np.asarray(arrays[name], dtype=float) for name in law.param_names]),
        run_sd=np.asarray(arrays["run_sd"], dtype=float),
        run_deviations=np.asarray(arrays["run_deviations"], dtype=float),
        method=method,
        n_failed=n_failed,
    )
