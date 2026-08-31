"""Point estimation of a scaling law and of the two variance components.

Fitting happens on **run-level means**, weighted by how precisely each mean is known. The optimization
exploits the fact that every law in this package is linear in its offset and amplitudes: for any candidate
exponent vector the remaining parameters are solved exactly by bounded linear least squares, so a coarse grid
over exponents gives excellent starting points and the nonlinear optimizer only has to polish them. This is
far more reliable at small sample sizes than random multistart, and it is fast enough to redo thousands of
times inside the bootstrap.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .laws import AMPLITUDE, EXPONENT, MAX_EXPONENT, LawInstance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .data import TargetObservations

#: Candidate exponent values used to seed the optimizer, spanning the empirically plausible range.
EXPONENT_GRID: tuple[float, ...] = (0.01, 0.03, 0.08, 0.2, 0.5, 1.2)

#: Cap on the number of grid combinations evaluated when a law has many exponents.
MAX_GRID_POINTS = 256

#: How many of the best grid starts are polished by the nonlinear optimizer.
MAX_REFINEMENTS = 5

#: Amplitude and offset bounds are this multiple of the observed spread of the target. They are
#: generous but finite on purpose. ``E + A * x**-alpha`` has a notorious flat ridge along which
#: ``alpha -> 0``, ``A -> +inf`` and ``E -> -inf`` while the curve barely changes; an unbounded box
#: lets the optimizer crawl down it indefinitely, wasting time and returning meaningless
#: parameters. A finite box stops the crawl and turns the degeneracy into a reported
#: ``parameter_at_bound`` warning, which is what a caller actually needs to know.
AMPLITUDE_BOUND_FACTOR = 50.0
OFFSET_BOUND_FACTOR = 50.0

#: Largest permitted ratio between the largest and smallest fitting weight. When the run-level
#: variance estimates to exactly zero -- which happens readily with two or three replicate runs --
#: the weights would otherwise be proportional to each run's evaluation count, which is precisely
#: the confusion between evaluation effort and training evidence this package exists to prevent.
MAX_WEIGHT_RATIO = 10.0

#: Replicate degrees of freedom required before the replicate-based run-variance estimate is
#: preferred over the residual-based one.
MIN_REPLICATE_DOF = 3

#: Relative distance from a bound at which a parameter counts as pinned to it.
BOUND_TOLERANCE = 1e-6

#: Optimizer convergence tolerances and per-start evaluation budget. The budget matters: on a
#: degenerate target the trust-region solver will otherwise take thousands of vanishing steps.
CONVERGENCE_TOLERANCE = 1e-10
MAX_EVALUATIONS_PER_PARAM = 60

#: A target whose values span less than this fraction of their own magnitude carries no scaling
#: information at all and is fit as a constant rather than handed to the optimizer.
CONSTANT_TARGET_TOLERANCE = 1e-10


@dataclasses.dataclass(frozen=True, slots=True)
class Bounds:
    """Box constraints on a parameter vector.

    Attributes:
        lower: Lower bounds, one per parameter.
        upper: Upper bounds, one per parameter.
    """

    lower: np.ndarray
    upper: np.ndarray

    def clip(self, params: np.ndarray) -> np.ndarray:
        """Project a parameter vector into the box."""
        return np.clip(params, self.lower, self.upper)

    def at_bound(self, params: np.ndarray, names: Sequence[str]) -> tuple[str, ...]:
        """Names of parameters sitting on a finite bound.

        Args:
            params: The fitted parameter vector.
            names: Parameter names, in parameter-vector order.

        Returns:
            The names of parameters within :data:`BOUND_TOLERANCE` of a finite bound.
        """
        pinned = []
        for value, low, high, name in zip(params, self.lower, self.upper, names, strict=True):
            width = max(abs(high - low), 1.0)
            at_low = np.isfinite(low) and abs(value - low) <= BOUND_TOLERANCE * width
            at_high = np.isfinite(high) and abs(value - high) <= BOUND_TOLERANCE * width
            if at_low or at_high:
                pinned.append(name)
        return tuple(pinned)


def parameter_bounds(law: LawInstance, y: np.ndarray, signed_amplitude: bool) -> Bounds:
    """Data-driven box constraints for a law's parameters.

    Exponents are always non-negative, so every term decays as scale grows. Loss amplitudes are
    non-negative too, which gives the conventional "irreducible floor plus decaying terms" reading.
    Metric amplitudes are free in sign so that an increasing metric can approach its asymptote from
    below. The remaining bounds are generous multiples of the observed spread; they exist to keep
    the optimizer in a numerically sane region, and a parameter that lands on one is reported.

    Args:
        law: The instantiated law.
        y: Observed target values, used to set the scale.
        signed_amplitude: Whether amplitudes may be negative.

    Returns:
        The :class:`Bounds` for this law and target.

    Examples:
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> bounds = parameter_bounds(law, np.array([1.0, 2.0, 3.0]), signed_amplitude=False)
        >>> bounds.lower.round(3)
        array([-99.,   0.,   0.,   0.,   0.])
        >>> bounds.at_bound(np.array([0.0, 0.0, 1.0, 0.5, 0.5]), law.param_names)
        ('A',)
    """
    y = np.asarray(y, dtype=float)
    spread = float(y.max() - y.min())
    # The box must admit an asymptote well below everything observed: a slowly scaling loss can sit
    # at 2.0 with a spread of 0.01 and still have its true floor near zero, which a box derived from
    # the spread alone would exclude. Tying the scale to the magnitude of the target as well keeps
    # that reachable.
    scale = max(spread, 0.1 * float(np.abs(y).mean()), 1e-8)
    lower, upper = [], []
    for kind in law.param_kinds:
        match kind:
            case _ if kind == EXPONENT:
                lower.append(0.0)
                upper.append(MAX_EXPONENT)
            case _ if kind == AMPLITUDE:
                lower.append(-AMPLITUDE_BOUND_FACTOR * scale if signed_amplitude else 0.0)
                upper.append(AMPLITUDE_BOUND_FACTOR * scale)
            case _:
                lower.append(float(y.min()) - OFFSET_BOUND_FACTOR * scale)
                upper.append(float(y.max()) + OFFSET_BOUND_FACTOR * scale)
    return Bounds(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _exponent_grid(n_exponents: int) -> np.ndarray:
    """Grid of candidate exponent vectors, subsampled when the full product is too large."""
    if n_exponents == 0:  # pragma: no cover - every law has at least one exponent
        return np.zeros((1, 0))
    total = len(EXPONENT_GRID) ** n_exponents
    if total <= MAX_GRID_POINTS:
        return np.array(list(itertools.product(EXPONENT_GRID, repeat=n_exponents)), dtype=float)
    rng = np.random.default_rng(0)
    return rng.choice(np.asarray(EXPONENT_GRID), size=(MAX_GRID_POINTS, n_exponents))


def _solve_linear(a: np.ndarray, b: np.ndarray, low: np.ndarray, high: np.ndarray):
    """Bounded linear least squares for the offset and amplitudes of an already-weighted system."""
    try:
        solution = lsq_linear(a, b, bounds=(low, high), method="bvls")
        return np.asarray(solution.x, dtype=float), float(solution.cost)
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - numerical fallback
        coefficients = np.clip(np.linalg.lstsq(a, b, rcond=None)[0], low, high)
        return coefficients, 0.5 * float(np.sum((a @ coefficients - b) ** 2))


class GridSeeder:
    """Reusable variable-projection restarts for a fixed design, weights and bounds.

    For each candidate exponent vector the offset and amplitudes are solved exactly, so each seed is
    already optimal in every direction except the exponents. The design matrices and their
    pseudo-inverses depend only on the exponent grid, the predictors and the weights -- none of
    which change across bootstrap draws -- so they are computed once here and reused thousands of
    times.

    Examples:
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> grid = np.array([[n, d] for n in (1.0, 10.0, 100.0) for d in (1.0, 10.0, 100.0)])
        >>> log_x = np.log(grid)
        >>> y = law.evaluate(np.array([0.5, 1.0, 2.0, 0.2, 0.5]), log_x)
        >>> bounds = parameter_bounds(law, y, signed_amplitude=False)
        >>> seeder = GridSeeder(law, log_x, np.ones(len(y)), bounds)
        >>> cost, seed = seeder.seeds(y)[0]
        >>> seed[3:].round(2)
        array([0.2, 0.5])
    """

    def __init__(self, law: LawInstance, log_x: np.ndarray, weights: np.ndarray, bounds: Bounds):
        """Precompute the weighted designs for every grid point.

        Args:
            law: The instantiated law.
            log_x: Log-normalized predictors, shape ``(n_obs, n_predictors)``.
            weights: Fitting weights, shape ``(n_obs,)``.
            bounds: Parameter box constraints.
        """
        self.law = law
        self.bounds = bounds
        self.sqrt_w = np.sqrt(np.asarray(weights, dtype=float))
        self.exponents = _exponent_grid(law.n_exponents)
        self.designs = [self.sqrt_w[:, None] * law.design(exponents, log_x) for exponents in self.exponents]
        self.pseudo_inverses = [np.linalg.pinv(design) for design in self.designs]
        self.low = bounds.lower[: law.n_linear]
        self.high = bounds.upper[: law.n_linear]

    def seeds(
        self, y: np.ndarray, n_seeds: int = MAX_REFINEMENTS, exact: bool = True
    ) -> list[tuple[float, np.ndarray]]:
        """Best starting points for these observations, ordered by increasing cost.

        Args:
            y: Observed values, shape ``(n_obs,)``.
            n_seeds: How many seeds to return.
            exact: Whether to solve the bounded linear problem exactly when the unconstrained
                solution violates a bound. Exact solves are used for the point estimate; the
                bootstrap uses the cheaper clipped solution, whose cost is an upper bound on the
                exact one and therefore still detects a genuinely better basin.

        Returns:
            Up to ``n_seeds`` ``(cost, params)`` pairs.
        """
        b = self.sqrt_w * y
        candidates: list[tuple[float, np.ndarray]] = []
        for i, design in enumerate(self.designs):
            linear = self.pseudo_inverses[i] @ b
            if np.any(linear < self.low) or np.any(linear > self.high):
                if exact:
                    linear, cost = _solve_linear(design, b, self.low, self.high)
                else:
                    linear = np.clip(linear, self.low, self.high)
                    cost = 0.5 * float(np.sum((design @ linear - b) ** 2))
            else:
                cost = 0.5 * float(np.sum((design @ linear - b) ** 2))
            candidates.append((cost, np.concatenate([linear, self.exponents[i]])))
        candidates.sort(key=lambda item: item[0])
        return [(cost, self.bounds.clip(params)) for cost, params in candidates[:n_seeds]]


@dataclasses.dataclass(frozen=True, slots=True)
class CurveFit:
    """The outcome of one nonlinear least-squares fit.

    Attributes:
        params: The fitted parameter vector.
        cost: Half the weighted sum of squared residuals at the optimum.
        success: Whether the optimizer reported convergence.
        n_function_evaluations: Total objective evaluations across all polished starts.
        message: The optimizer's termination message.
        n_starts: How many starting points were polished.
    """

    params: np.ndarray
    cost: float
    success: bool
    n_function_evaluations: int
    message: str
    n_starts: int


def fit_curve(
    law: LawInstance,
    log_x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bounds: Bounds,
    starts: Sequence[np.ndarray] | None = None,
    seeder: GridSeeder | None = None,
) -> CurveFit:
    """Fit a law to run-level observations by weighted nonlinear least squares.

    Args:
        law: The instantiated law.
        log_x: Log-normalized predictors, shape ``(n_obs, n_predictors)``.
        y: Observed values, shape ``(n_obs,)``.
        weights: Fitting weights, shape ``(n_obs,)``.
        bounds: Parameter box constraints.
        starts: Explicit starting points. Defaults to the best :class:`GridSeeder` seeds; pass a
            single previously fitted vector to warm-start a bootstrap refit.
        seeder: A reusable seeder to draw default starting points from.

    Returns:
        The best :class:`CurveFit` across all starting points.

    Examples:
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> grid = np.array([[n, d] for n in (1.0, 10.0, 100.0) for d in (1.0, 10.0, 100.0)])
        >>> log_x = np.log(grid)
        >>> truth = np.array([0.5, 1.0, 2.0, 0.3, 0.6])
        >>> y = law.evaluate(truth, log_x)
        >>> bounds = parameter_bounds(law, y, signed_amplitude=False)
        >>> fit = fit_curve(law, log_x, y, np.ones(len(y)), bounds)
        >>> fit.params.round(3)
        array([0.5, 1. , 2. , 0.3, 0.6])
        >>> fit.success
        True
    """
    sqrt_w = np.sqrt(weights)
    if starts is None:
        seeder = seeder or GridSeeder(law, log_x, weights, bounds)
        starts = [params for _, params in seeder.seeds(y)]
    starts = list(starts)

    def residual(params: np.ndarray) -> np.ndarray:
        return sqrt_w * (law.evaluate(params, log_x) - y)

    def jacobian(params: np.ndarray) -> np.ndarray:
        return sqrt_w[:, None] * law.jacobian(params, log_x)

    best: Any = None
    evaluations = 0
    for start in starts:
        result = least_squares(
            residual,
            bounds.clip(np.asarray(start, dtype=float)),
            jac=jacobian,
            bounds=(bounds.lower, bounds.upper),
            method="trf",
            # The parameters are already comparably scaled by construction -- predictors are
            # normalized to a geometric mean of one, amplitudes sit on the target's own scale, and
            # exponents are order one -- so scipy's Jacobian-based rescaling buys nothing here and
            # costs roughly three times the runtime of the whole fit.
            x_scale=1.0,
            ftol=CONVERGENCE_TOLERANCE,
            xtol=CONVERGENCE_TOLERANCE,
            gtol=CONVERGENCE_TOLERANCE,
            max_nfev=MAX_EVALUATIONS_PER_PARAM * law.n_params,
        )
        evaluations += int(result.nfev)
        if best is None or result.cost < best.cost:
            best = result
    return CurveFit(
        params=np.asarray(best.x, dtype=float),
        cost=float(best.cost),
        success=bool(best.success),
        n_function_evaluations=evaluations,
        message=str(best.message),
        n_starts=len(starts),
    )


def is_constant(y: np.ndarray) -> bool:
    """Whether a target carries no variation to explain.

    A target that never changes -- an AUROC pinned at 1.0, a metric that simply does not scale --
    has no scaling law to find. Detecting it up front matters for more than tidiness: handing a
    flat target to a bounded trust-region solver produces thousands of vanishing steps per fit,
    which is slow enough to look like a hang once multiplied by the bootstrap.

    Args:
        y: Observed values.

    Returns:
        Whether the values are constant to within :data:`CONSTANT_TARGET_TOLERANCE`.

    Examples:
        >>> is_constant(np.array([0.8, 0.8, 0.8]))
        True
        >>> is_constant(np.array([0.8, 0.81, 0.8]))
        False
    """
    y = np.asarray(y, dtype=float)
    return bool(np.ptp(y) <= CONSTANT_TARGET_TOLERANCE * max(1.0, float(np.abs(y).max())))


def constant_fit(law: LawInstance, y: np.ndarray) -> CurveFit:
    """The exact fit for a target with no variation: the offset alone, every other term zero.

    Args:
        law: The instantiated law.
        y: Observed values.

    Returns:
        The trivial :class:`CurveFit`.

    Examples:
        >>> from .laws import build_law
        >>> law = build_law("separable-power", ["model_size__n"], ["dataset_size__d"])
        >>> constant_fit(law, np.array([0.8, 0.8])).params
        array([0.8, 0. , 0. , 0. , 0. ])
    """
    params = np.zeros(law.n_params, dtype=float)
    params[0] = float(np.mean(y))
    return CurveFit(
        params=params,
        cost=0.0,
        success=True,
        n_function_evaluations=0,
        message="Target is constant; fit analytically as an offset.",
        n_starts=0,
    )


def leverage(law: LawInstance, params: np.ndarray, log_x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-observation leverage of the linearized fit at ``params``.

    Args:
        law: The instantiated law.
        params: The fitted parameter vector.
        log_x: Log-normalized predictors.
        weights: Fitting weights.

    Returns:
        Leverages ``h_r`` in ``[0, 1]``, one per observation. These say how much each observation
        pulls the curve toward itself, and hence how much fitting shrinks its own residual.
    """
    a = np.sqrt(weights)[:, None] * law.jacobian(params, log_x)
    hat = a @ np.linalg.pinv(a.T @ a) @ a.T
    return np.clip(np.diag(hat), 0.0, 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class VarianceComponents:
    """The two variance components of the statistical model.

    Attributes:
        eval_var: Pooled within-run evaluation variance.
        eval_var_independent: The part of ``eval_var`` that averages away across runs.
        run_var: The run-to-run training variance actually used.
        run_var_replicate: The replicate-based estimate, or ``None`` if not estimable.
        run_var_residual: The residual-based estimate, or ``None`` if not estimable.
        replicate_dof: Degrees of freedom behind the replicate-based estimate.
        residual_dof: Degrees of freedom behind the residual-based estimate.
        source: ``"replicates"``, ``"residual"``, or ``"none"``.
    """

    eval_var: float
    eval_var_independent: float
    run_var: float
    run_var_replicate: float | None
    run_var_residual: float | None
    replicate_dof: int
    residual_dof: int
    source: str

    @property
    def run_sd(self) -> float:
        """Standard deviation of the run-to-run training variation."""
        return float(np.sqrt(self.run_var))

    @property
    def eval_sd(self) -> float:
        """Standard deviation of the finite-evaluation-set noise on a single evaluation row."""
        return float(np.sqrt(self.eval_var))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


def replicate_run_variance(
    observations: TargetObservations, eval_var_independent: float
) -> tuple[float | None, int]:
    """Estimate run-to-run variance from independently trained models at the same configuration.

    Within one configuration, the spread of run means reflects both genuine run-to-run variation and
    the residual evaluation noise left in each mean. The latter is subtracted off exactly:
    for a configuration with ``n_c`` runs and per-run mean variances ``v_r``, the expected
    within-configuration sum of squares is ``(n_c - 1) * run_var + (1 - 1/n_c) * sum_r v_r``.

    Args:
        observations: Run-level observations for one target.
        eval_var_independent: The evaluation variance that averages away across runs.

    Returns:
        The pooled estimate (floored at zero) and its degrees of freedom, or ``(None, 0)`` when no
        configuration has replicate runs.

    Examples:
        A configuration with two runs whose means differ by 0.2, and no evaluation noise, implies a
        run-level variance of ``0.02``:

        >>> from .data import TargetObservations
        >>> observations = TargetObservations(
        ...     target="t",
        ...     run_ids=("r1", "r2"),
        ...     config_index=np.array([0, 0]),
        ...     mean=np.array([1.0, 1.2]),
        ...     n_eval=np.array([5, 5]),
        ...     within_ss=0.0,
        ...     within_dof=8,
        ...     eval_pair_correlation=None,
        ...     n_shared_pairs=0,
        ... )
        >>> estimate, dof = replicate_run_variance(observations, 0.0)
        >>> round(estimate, 6), dof
        (0.02, 1)
    """
    per_run_var = eval_var_independent / np.maximum(observations.n_eval, 1)
    total_ss, correction, dof = 0.0, 0.0, 0
    for config in np.unique(observations.config_index):
        members = np.flatnonzero(observations.config_index == config)
        n_c = members.size
        if n_c < 2:
            continue
        means = observations.mean[members]
        total_ss += float(((means - means.mean()) ** 2).sum())
        correction += (1.0 - 1.0 / n_c) * float(per_run_var[members].sum())
        dof += n_c - 1
    if dof == 0:
        return None, 0
    return max(0.0, (total_ss - correction) / dof), dof


def residual_run_variance(
    residuals: np.ndarray, leverages: np.ndarray, per_run_var: np.ndarray, n_params: int
) -> tuple[float | None, int]:
    """Estimate run-to-run variance from the residuals of the fitted curve.

    This is the fallback used when no configuration has replicate runs. It cannot separate genuine
    run-to-run variation from misspecification of the scaling law, so it is biased *upward* whenever
    the law does not fit -- which is the conservative direction, and is reported as such.

    Args:
        residuals: Run-level residuals ``ybar_r - f(x_r)``.
        leverages: Per-observation leverages from :func:`leverage`.
        per_run_var: Evaluation-noise variance of each run mean.
        n_params: Number of fitted parameters.

    Returns:
        The estimate (floored at zero) and its residual degrees of freedom, or ``(None, 0)`` when
        there are no residual degrees of freedom left.
    """
    dof = residuals.size - n_params
    if dof <= 0:
        return None, 0
    expected_eval = float(((1.0 - leverages) * per_run_var).sum())
    return max(0.0, (float((residuals**2).sum()) - expected_eval) / dof), int(dof)


def compute_weights(run_var: float, per_run_var: np.ndarray) -> np.ndarray:
    """Fitting weights, normalized to mean one and range-limited.

    Each run mean has variance ``run_var + eval_var_independent / n_r``, so its weight is the
    reciprocal. The ratio between the largest and smallest weight is capped at
    :data:`MAX_WEIGHT_RATIO` so that a run with an unusually large number of evaluation rows cannot
    dominate the fit when the run-level variance happens to estimate to zero.

    Args:
        run_var: Estimated run-to-run variance.
        per_run_var: Evaluation-noise variance of each run mean.

    Returns:
        Weights of shape ``per_run_var.shape``, averaging to one.

    Examples:
        >>> compute_weights(0.0, np.array([0.0, 0.0]))
        array([1., 1.])
        >>> compute_weights(1.0, np.array([0.0, 1.0])).round(4)
        array([1.3333, 0.6667])
    """
    total = run_var + per_run_var
    if not np.any(total > 0):
        return np.ones_like(per_run_var, dtype=float)
    total = np.maximum(total, total[total > 0].max() / MAX_WEIGHT_RATIO)
    weights = 1.0 / total
    return weights / weights.mean()
