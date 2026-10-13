"""Point estimation of a scaling law and of the two variance components.

Fitting happens on **run-level means**, weighted by how precisely each mean is known. The optimization
exploits variable projection: for any candidate nonlinear parameter vector, the offset and amplitudes are
solved exactly by bounded linear least squares. A small, parameter-kind-aware grid gives stable starting
points, and the nonlinear optimizer only has to polish the best few. This remains far more reliable at small
sample sizes than unconstrained random multistart and is fast enough to repeat in the bootstrap.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .laws import AMPLITUDE, EXPONENT, MAX_EXPONENT, SCALE, SHIFT, LawInstance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .data import TargetObservations

#: Candidate exponent values used to seed the optimizer.
EXPONENT_GRID: tuple[float, ...] = (0.01, 0.03, 0.08, 0.2, 0.5, 1.2)

#: Candidate normalized additive shifts. Zero recovers an ordinary power law.
SHIFT_GRID: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

#: Candidate positive scale ratios, centered on the normalized reference scale.
SCALE_GRID: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

#: Conservative finite bounds for normalized shifts and ratios.
MAX_SHIFT = 10.0
MAX_SCALE = 100.0

#: Cap on nonlinear grid combinations evaluated for a law.
MAX_GRID_POINTS = 256

#: How many of the best grid starts are polished by the nonlinear optimizer.
MAX_REFINEMENTS = 5

#: Amplitude and offset bounds are this multiple of the observed spread of the target.
AMPLITUDE_BOUND_FACTOR = 50.0
OFFSET_BOUND_FACTOR = 50.0

#: Largest permitted ratio between the largest and smallest fitting weight.
MAX_WEIGHT_RATIO = 10.0

#: Replicate degrees of freedom required before the replicate-based run-variance estimate is
#: preferred over the residual-based one.
MIN_REPLICATE_DOF = 3

#: Relative distance from a bound at which a parameter counts as pinned to it.
BOUND_TOLERANCE = 1e-6

#: Optimizer convergence tolerances and per-start evaluation budget.
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
        """Names of parameters sitting on a finite bound."""
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

    Exponents, shifts, and positive scale ratios are non-negative. Loss amplitudes are non-negative; metric
    amplitudes may be signed. Offsets retain the same generous data-driven box as before.
    """
    y = np.asarray(y, dtype=float)
    spread = float(y.max() - y.min())
    scale = max(spread, 0.1 * float(np.abs(y).mean()), 1e-8)
    lower, upper = [], []
    for kind in law.param_kinds:
        match kind:
            case _ if kind == EXPONENT:
                lower.append(0.0)
                upper.append(MAX_EXPONENT)
            case _ if kind == SHIFT:
                lower.append(0.0)
                upper.append(MAX_SHIFT)
            case _ if kind == SCALE:
                lower.append(0.0)
                upper.append(MAX_SCALE)
            case _ if kind == AMPLITUDE:
                lower.append(-AMPLITUDE_BOUND_FACTOR * scale if signed_amplitude else 0.0)
                upper.append(AMPLITUDE_BOUND_FACTOR * scale)
            case _:
                lower.append(float(y.min()) - OFFSET_BOUND_FACTOR * scale)
                upper.append(float(y.max()) + OFFSET_BOUND_FACTOR * scale)
    return Bounds(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _grid_for_kind(kind: str) -> tuple[float, ...]:
    """Candidate seed values for one nonlinear parameter kind."""
    if kind == EXPONENT:
        return EXPONENT_GRID
    if kind == SHIFT:
        return SHIFT_GRID
    if kind == SCALE:
        return SCALE_GRID
    raise ValueError(f"No nonlinear seed grid for parameter kind {kind!r}")


def _nonlinear_grid(law: LawInstance) -> np.ndarray:
    """Candidate nonlinear vectors, deterministically subsampled when the product is too large."""
    grids = [_grid_for_kind(kind) for kind in law.nonlinear_kinds]
    if not grids:  # pragma: no cover - built-in laws all have nonlinear parameters
        return np.zeros((1, 0))
    total = int(np.prod([len(grid) for grid in grids]))
    if total <= MAX_GRID_POINTS:
        return np.asarray(list(itertools.product(*grids)), dtype=float)
    rng = np.random.default_rng(0)
    return np.column_stack(
        [rng.choice(np.asarray(grid), size=MAX_GRID_POINTS, replace=True) for grid in grids]
    )


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

    For each candidate nonlinear vector, the linear coefficients are solved exactly. The resulting design
    matrices and pseudo-inverses are reusable across bootstrap draws because predictors and fitting weights do
    not change.
    """

    def __init__(self, law: LawInstance, log_x: np.ndarray, weights: np.ndarray, bounds: Bounds):
        """Precompute the weighted designs for every nonlinear grid point."""
        self.law = law
        self.bounds = bounds
        self.sqrt_w = np.sqrt(np.asarray(weights, dtype=float))
        self.nonlinear = _nonlinear_grid(law)
        self.designs = [self.sqrt_w[:, None] * law.design(nonlinear, log_x) for nonlinear in self.nonlinear]
        self.pseudo_inverses = [np.linalg.pinv(design) for design in self.designs]
        self.low = bounds.lower[: law.n_linear]
        self.high = bounds.upper[: law.n_linear]

    def seeds(
        self, y: np.ndarray, n_seeds: int = MAX_REFINEMENTS, exact: bool = True
    ) -> list[tuple[float, np.ndarray]]:
        """Best starting points for these observations, ordered by increasing cost."""
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
            candidates.append((cost, np.concatenate([linear, self.nonlinear[i]])))
        candidates.sort(key=lambda item: item[0])
        return [(cost, self.bounds.clip(params)) for cost, params in candidates[:n_seeds]]


@dataclasses.dataclass(frozen=True, slots=True)
class CurveFit:
    """The outcome of one nonlinear least-squares fit."""

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
    """Fit a law to run-level observations by weighted nonlinear least squares."""
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
    """Whether a target carries no variation to explain."""
    y = np.asarray(y, dtype=float)
    return bool(np.ptp(y) <= CONSTANT_TARGET_TOLERANCE * max(1.0, float(np.abs(y).max())))


def constant_fit(law: LawInstance, y: np.ndarray) -> CurveFit:
    """The exact fit for a target with no variation: offset alone, every other term zero."""
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
    """Per-observation leverage of the linearized fit at ``params``."""
    a = np.sqrt(weights)[:, None] * law.jacobian(params, log_x)
    hat = a @ np.linalg.pinv(a.T @ a) @ a.T
    return np.clip(np.diag(hat), 0.0, 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class VarianceComponents:
    """The two variance components of the statistical model."""

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
    """Estimate run-to-run variance from independently trained models at the same configuration."""
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
    """Estimate run-to-run variance from residuals of the fitted curve."""
    dof = residuals.size - n_params
    if dof <= 0:
        return None, 0
    expected_eval = float(((1.0 - leverages) * per_run_var).sum())
    return max(0.0, (float((residuals**2).sum()) - expected_eval) / dof), int(dof)


def compute_weights(run_var: float, per_run_var: np.ndarray) -> np.ndarray:
    """Fitting weights, normalized to mean one and range-limited."""
    total = run_var + per_run_var
    if not np.any(total > 0):
        return np.ones_like(per_run_var, dtype=float)
    total = np.maximum(total, total[total > 0].max() / MAX_WEIGHT_RATIO)
    weights = 1.0 / total
    return weights / weights.mean()
