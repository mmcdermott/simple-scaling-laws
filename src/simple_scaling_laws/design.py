"""Planning experiments: which runs to train, before any exist and once some do.

Two tools, for two situations.

:func:`isoflop_grid` lays out a whole experiment in advance: sweeps of model size along curves of equal
cost, at a few budgets, centred on a guess at the optimum. The same runs serve both estimators of the
compute-optimal frontier. Along each curve they give an isoFLOP profile
(:func:`~simple_scaling_laws.isoflop.fit_isoflop`); across curves they form a grid rotated in
``(log N, log D)`` that a joint scaling law can be fit to (:func:`~simple_scaling_laws.fit`). Two facts
shape it:

* **Width along a curve is free.** Every point on a curve of equal cost costs the same, so sweeping
  wider spends nothing more; it only moves runs toward the lopsided corners -- a huge model on little
  data, a tiny one on a lot. Scored by :func:`score_design` on four budgets with six sizes each, wider
  sweeps kept improving the law's frontier (RMSE of ``log10 N*`` a hundred times past the largest
  budget fell from 0.008-0.012 at ±0.9 decades to 0.003-0.007 at ±1.8), while the isoFLOP profiles did
  best around ±1.2 and lost accuracy beyond, where a cubic is asked to describe too much of the valley.
  Narrow sweeps also risk missing an optimum that the centre guessed wrong: with ±0.6 decades and a
  centre guessed for a different law, the profiles' error was a third of a decade. The default is ±1.2.
* **Spread in budget versus spread in ratio.** Across the design, the correlation between ``log N`` and
  ``log D`` is ``(var(log C) - var(log N/D)) / (var(log C) + var(log N/D))``: many budgets with narrow
  sweeps make the two predictors nearly collinear, which the package reports and which leaves the law's
  two terms hard to tell apart (:func:`design_correlation`).

:func:`suggest_runs` picks the next runs once some exist, to shrink the uncertainty in the quantities
you care about -- the optimal model size at the budgets you will eventually spend -- as cheaply as
possible. See its docstring for the approximation it rests on and what was measured.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from .model import PredictionError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .cost import CostModel
    from .isoflop import IsoflopFit
    from .model import ScalingLawModel

#: Default predictor columns of a planned design.
DESIGN_PREDICTORS: tuple[str, str] = ("model_size__n_params", "dataset_size__n_tokens")


def _centres(
    budgets: np.ndarray, cost: CostModel, centre: float | Sequence[float] | ScalingLawModel | IsoflopFit
) -> np.ndarray:
    """The model size each sweep is centred on: from a data-to-model ratio, explicit sizes, or a fit."""
    from .isoflop import IsoflopFit
    from .model import ScalingLawModel

    if isinstance(centre, ScalingLawModel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frontier = centre.optimal_allocation(list(budgets), cost)
        return frontier[f"{centre.law.model_size[0]}__median"].to_numpy().astype(float)
    if isinstance(centre, IsoflopFit):
        return centre.predict(list(budgets))[f"{centre.model_size}__median"].to_numpy().astype(float)
    if np.ndim(centre) == 0:
        ratio = float(centre)
        if not ratio > 0:
            raise PredictionError(f"A data-to-model ratio must be positive, got {ratio}")
        # Solve D / N = ratio on each iso-cost curve, by bisection in log N.
        out = []
        for budget in budgets:
            low, high = np.log(1e-6), np.log(1e30)
            for _ in range(200):
                middle = 0.5 * (low + high)
                n = np.exp(middle)
                if float(cost.dataset_size_for(budget, n)) / n > ratio:
                    low = middle
                else:
                    high = middle
            out.append(np.exp(0.5 * (low + high)))
        return np.array(out)
    sizes = np.asarray(centre, dtype=float)
    if sizes.shape != budgets.shape:
        raise PredictionError(f"Expected one centre per budget ({budgets.size}), got {sizes.size}")
    return sizes


def isoflop_grid(
    budgets: Sequence[float],
    cost: CostModel,
    centre: float | Sequence[float] | ScalingLawModel | IsoflopFit,
    points_per_budget: int = 6,
    half_width: tuple[float, float] = (1.2, 1.2),
    centre_replicates: int = 2,
    predictors: tuple[str, str] = DESIGN_PREDICTORS,
) -> pl.DataFrame:
    """Sweeps of model size along curves of equal cost: one design for isoFLOP profiles and a joint law.

    Args:
        budgets: The budgets to sweep, in the units of ``cost``.
        cost: The cost law.
        centre: Where each sweep is centred: a guessed data-to-model ratio ``D / N`` at the optimum (for
            language models trained on tokens, about 20 is the published Chinchilla figure); explicit
            model sizes, one per budget; or a fit -- a scaling law's frontier or an isoFLOP line -- from
            runs already made.
        points_per_budget: Model sizes per sweep. Five is the least a cubic isoFLOP profile can be
            judged with; six leaves room for one to fail.
        half_width: Decades of model size either side of the centre, at the smallest and the largest
            budget; budgets in between are interpolated in ``log C``. See the module docstring for
            what was measured; the default, ±1.2 decades throughout, suited both frontiers.
        centre_replicates: Independent runs at each sweep's centre. Replicates are what estimate
            run-to-run variance directly; elsewhere one run per point.
        predictors: Names of the model-size and dataset-size columns.

    Returns:
        One row per planned configuration: ``budget``, the two predictors, ``runs`` (how many independent
        runs to train there) and ``cost`` (of one run).

    Raises:
        PredictionError: If an argument is invalid.

    Examples:
        >>> from simple_scaling_laws.cost import cost_model
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> grid = isoflop_grid([1e17, 1e19], chinchilla, centre=20.0, points_per_budget=5)
        >>> summary = grid.group_by("budget", maintain_order=True).agg(
        ...     pl.len().alias("sizes"), pl.col("runs").sum(), pl.col("model_size__n_params").min()
        ... )
        >>> [(f"{b:.0e}", sizes, runs, f"{smallest:.2g}") for b, sizes, runs, smallest in summary.rows()]
        [('1e+17', 5, 6, '1.8e+06'), ('1e+19', 5, 6, '1.8e+07')]

        Every point spends its budget exactly:

        >>> bool(np.allclose(grid["cost"], grid["budget"]))
        True
    """
    budgets_array = np.sort(np.asarray([float(b) for b in budgets]))
    if budgets_array.size == 0 or not np.all(np.isfinite(budgets_array) & (budgets_array > 0)):
        raise PredictionError(f"Budgets must be positive and finite, got {list(budgets)}")
    if points_per_budget < 2:
        raise PredictionError(f"points_per_budget must be at least 2, got {points_per_budget}")
    if min(half_width) <= 0:
        raise PredictionError(f"half_width must be positive, got {half_width}")
    centres = _centres(budgets_array, cost, centre)
    log_c = np.log(budgets_array)
    position = (log_c - log_c[0]) / (log_c[-1] - log_c[0]) if budgets_array.size > 1 else np.zeros(1)
    widths = half_width[0] + (half_width[1] - half_width[0]) * position
    rows: list[dict[str, Any]] = []
    for budget, middle, width in zip(budgets_array, centres, widths, strict=True):
        offsets = np.linspace(-width, width, points_per_budget)
        nearest = int(np.argmin(np.abs(offsets)))
        for i, offset in enumerate(offsets):
            n = float(middle * 10.0**offset)
            d = float(cost.dataset_size_for(budget, n))
            rows.append(
                {
                    "budget": float(budget),
                    predictors[0]: n,
                    predictors[1]: d,
                    "runs": centre_replicates if i == nearest else 1,
                    "cost": float(cost(n, d)),
                }
            )
    return pl.DataFrame(rows)


def design_correlation(design: pl.DataFrame, predictors: tuple[str, str] = DESIGN_PREDICTORS) -> float:
    """Correlation of ``log N`` and ``log D`` across a design, counting replicate runs.

    Above 0.95 the package reports the predictors as collinear once the runs are fit.

    Examples:
        >>> from simple_scaling_laws.cost import cost_model
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> wide = isoflop_grid([1e15, 1e17, 1e19], chinchilla, centre=20.0, half_width=(1.2, 0.6))
        >>> narrow = isoflop_grid([1e15, 1e17, 1e19], chinchilla, centre=20.0, half_width=(0.2, 0.2))
        >>> round(design_correlation(wide), 2), round(design_correlation(narrow), 2)
        (0.31, 0.95)
    """
    repeat = design["runs"].to_numpy().astype(int)
    log_n = np.repeat(np.log(design[predictors[0]].to_numpy()), repeat)
    log_d = np.repeat(np.log(design[predictors[1]].to_numpy()), repeat)
    return float(np.corrcoef(log_n, log_d)[0, 1])


def _target_draws(
    model: ScalingLawModel, cost: CostModel, budgets: Sequence[float], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """``log10 N*`` at each budget, per draw, and which draws have an interior optimum at all of them."""
    from .allocation import _optimum_draws

    stars = [_optimum_draws(model, cost, float(budget), target) for budget in budgets]
    interior = ~np.any([boundary for *_, boundary in stars], axis=0)
    return np.log10(np.column_stack([star[0] for star in stars])), interior


def _candidates(
    model: ScalingLawModel, cost: CostModel, budgets: Sequence[float], target: str, points: int, spread: float
) -> np.ndarray:
    """Points along each candidate budget's iso-cost curve, spanning where its optimum might be."""
    from .allocation import _optimum_draws

    n_col = model.law.model_size[0]
    observed = model.observed_domain[n_col]
    rows = []
    for budget in budgets:
        n_star = _optimum_draws(model, cost, float(budget), target)[0]
        finite = n_star[np.isfinite(n_star)]
        if finite.size:
            low, high = np.quantile(finite, 0.01), np.quantile(finite, 0.99)
        else:
            low, high = observed["min"], observed["max"]
        for n in np.exp(
            np.linspace(np.log(low) - spread * np.log(10), np.log(high) + spread * np.log(10), points)
        ):
            rows.append((float(budget), float(n), float(cost.dataset_size_for(budget, n))))
    return np.array(rows)


def suggest_runs(
    model: ScalingLawModel,
    target_budgets: Sequence[float],
    cost: CostModel,
    n: int = 5,
    candidate_budgets: Sequence[float] | None = None,
    target: str | None = None,
    evaluations_per_run: int = 1,
    per_cost: bool = True,
    points: int = 31,
    spread: float = 0.6,
    max_per_point: int | None = None,
) -> pl.DataFrame:
    """The next runs to train, chosen to shrink the uncertainty in the optimal model size where it matters.

    **The criterion.** The quantities of interest are ``log N*`` at each of ``target_budgets`` -- usually
    budgets larger than anything you can afford to run. Each candidate run is a point on the iso-cost
    curve of one of ``candidate_budgets``. The fit's bootstrap draws are treated as an approximate
    posterior over the law, and the draws' joint spread of the targets and of the law's value at every
    candidate as a Gaussian. A run at candidate ``i`` observes the law there plus noise of variance
    ``s**2`` -- the fit's run-to-run variance plus the evaluation variance of ``evaluations_per_run``
    evaluations -- and conditioning on it shrinks the variance of target ``t`` by
    ``Cov(t, f_i)**2 / (Var(f_i) + s**2)``. The run with the largest total relative reduction (per unit
    cost, by default) is chosen, the covariance is conditioned on it, and the next is chosen the same way
    -- the standard greedy approximation to a Bayesian batch design. A point can be chosen more than once,
    which is a replicate. For a single target quantity the criterion often puts a whole batch on one or
    two points: the optimal design for one quantity has few support points (Elfving's theorem), which is
    efficient if the law is right and fragile if it is not. ``max_per_point`` caps it, trading some
    efficiency for spread.

    **What that rests on.** Three approximations, stated so you can judge them: that the bootstrap draws
    stand in for a posterior; that conditioning is close enough to linear-Gaussian over one batch; and
    that the law is right, since its draws are all the criterion sees. The last is the one that matters.
    Refit and ask again after each batch, rather than asking for many runs at once, and keep the isoFLOP
    check (:func:`~simple_scaling_laws.isoflop.fit_isoflop`) possible by starting from an
    :func:`isoflop_grid`; suggested runs alone rarely give each budget the five sizes a profile needs.

    **What was measured.** Simulated 60 times: a pilot of two cheap budgets swept wide, then the same
    extra compute spent either on an :func:`isoflop_grid` at two larger budgets or on three rounds of
    suggested runs at those budgets, each followed by a refit. Scored on ``N*`` a hundred times past the
    largest budget, the suggested runs halved the error (RMSE 0.005 against 0.009 decades at noise 0.02;
    0.001 against 0.002 at noise 0.005) with intervals half as wide, for no more compute. Coverage of the
    95% interval stayed at 97% at the higher noise and was 90% at the lower; with data from a Skaling law
    the separable fit could not represent, suggested runs still had the lower error and full coverage.

    Args:
        model: The fit so far, over one model-size and one dataset-size predictor.
        target_budgets: Budgets at which ``N*`` should be known precisely.
        cost: The cost law.
        n: Runs to suggest.
        candidate_budgets: Budgets you are willing to run at; by default ``target_budgets``.
        target: The target to optimize; by default the primary target.
        evaluations_per_run: Evaluations each new run will be scored on.
        per_cost: Rank candidates by variance removed per unit cost (default) rather than in total.
        points: Candidate points along each candidate budget's curve.
        spread: Decades beyond the range of the draws' optima that candidates extend.
        max_per_point: At most this many suggested runs at any one candidate point; by default no cap.

    Returns:
        One row per suggested run, in the order chosen: ``budget``, the two predictors, ``cost``, and
        ``remaining_variance``, the fraction of the targets' (summed, relative) variance expected to be left
        once this run and the ones before it are in.

    Raises:
        PredictionError: If the law is not over one predictor of each role, or too few draws have an
            interior optimum at the target budgets to estimate anything.

    Examples:
        >>> from simple_scaling_laws import fit
        >>> from simple_scaling_laws.cost import cost_model
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> pilot = isoflop_grid([1e15, 1e16], chinchilla, centre=20.0, half_width=(1.2, 1.0))
        >>> runs = simulate_runs(
        ...     {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}},
        ...     configurations=list(zip(pilot["model_size__n_params"], pilot["dataset_size__n_tokens"])),
        ...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
        ...     run_sd=0.02,
        ...     eval_sd=0.03,
        ...     seed=0,
        ... )
        >>> model = fit(runs, n_draws=200, seed=0)
        >>> plan = suggest_runs(model, [1e20], chinchilla, n=4, candidate_budgets=[1e17, 1e18])
        >>> plan.columns
        ['budget', 'model_size__n_params', 'dataset_size__n_tokens', 'cost', 'remaining_variance']
        >>> bool(plan["remaining_variance"].is_sorted(descending=True))
        True
    """
    from .allocation import _predictor_pair

    n_col, d_col = _predictor_pair(model)
    target = target or model.primary_target
    if n < 1:
        raise PredictionError(f"n must be at least 1, got {n}")
    targets, interior = _target_draws(model, cost, target_budgets, target)
    if interior.sum() < 20:
        raise PredictionError(
            f"Only {int(interior.sum())} draws have an interior optimum at every target budget, too few to "
            "say where uncertainty lies; the law does not determine the optimum there"
        )
    if interior.mean() < 0.9:
        warnings.warn(
            f"[no_interior_optimum] {1 - interior.mean():.0%} of draws have no interior optimum at some "
            "target budget and were left out of the criterion.",
            UserWarning,
            stacklevel=2,
        )
    candidates = _candidates(model, cost, candidate_budgets or target_budgets, target, points, spread)
    reference = np.array([model.observed_domain[p]["reference"] for p in model.fitted_predictors])
    log_x = np.log(candidates[:, 1:]) - np.log(reference)
    values = model.law.evaluate_many(model.draws[target].params[interior], log_x)
    components = model.fits[target].variance_components
    noise = float(components["run_var"]) + float(components["eval_var_independent"]) / evaluations_per_run
    noise = max(noise, 1e-12)

    joint = np.cov(np.column_stack([targets[interior], values]), rowvar=False)
    n_targets = targets.shape[1]
    initial = np.diag(joint)[:n_targets].copy()
    scale = candidates[:, 0] if per_cost else np.ones(len(candidates))
    taken = np.zeros(len(candidates), dtype=int)
    rows = []
    for _ in range(n):
        cross = joint[:n_targets, n_targets:]
        reduction = (cross**2 / (np.diag(joint)[n_targets:] + noise)[None, :] / initial[:, None]).sum(axis=0)
        score = reduction / scale
        if max_per_point is not None:
            score = np.where(taken < max_per_point, score, -np.inf)
            if not np.isfinite(score).any():
                break
        i = int(np.argmax(score))
        taken[i] += 1
        column = joint[:, n_targets + i].copy()
        joint = joint - np.outer(column, column) / (column[n_targets + i] + noise)
        budget, n_value, d_value = candidates[i]
        rows.append(
            {
                "budget": float(budget),
                n_col: float(n_value),
                d_col: float(d_value),
                "cost": float(cost(n_value, d_value)),
                "remaining_variance": float((np.diag(joint)[:n_targets] / initial).mean()),
            }
        )
    return pl.DataFrame(rows)


def _true_log10_optimum(
    truth: dict, law: Any, reference: np.ndarray, cost: CostModel, budget: float
) -> float:
    vector = np.array([truth[name] for name in law.param_names])

    def loss(log_n: float) -> float:
        size = np.exp(log_n)
        point = np.log(np.array([[size, float(cost.dataset_size_for(budget, size))]])) - np.log(reference)
        return float(law.evaluate(vector, point)[0])

    result = minimize_scalar(loss, bounds=(np.log(1e-3), np.log(1e30)), method="bounded")
    return float(result.x / np.log(10.0))


def score_design(
    design: pl.DataFrame,
    truths: Sequence[dict[str, float]],
    reference: Sequence[float],
    cost: CostModel,
    target_budgets: Sequence[float],
    law: str = "separable-power",
    run_sd: float = 0.02,
    eval_sd: float = 0.03,
    evaluations_per_run: int = 4,
    replications: int = 20,
    n_draws: int = 200,
    seed: int = 0,
    predictors: tuple[str, str] = DESIGN_PREDICTORS,
) -> pl.DataFrame:
    """How well a planned design would pin down the optimum, judged by simulating it before running it.

    For each truth -- a plausible law, stated in raw units relative to ``reference`` so that every design
    is judged against the same one -- the design's runs are simulated ``replications`` times with the given
    noise, and both frontiers are fit: the law's (:func:`~simple_scaling_laws.fit` then
    :func:`~simple_scaling_laws.allocation.optimal_allocation`) and the isoFLOP profiles'
    (:func:`~simple_scaling_laws.isoflop.fit_isoflop`). Compare candidate designs by their scores under
    the same truths, noise and seed.

    Args:
        design: A plan from :func:`isoflop_grid` (any frame with the two predictors, ``budget`` and
            ``runs``).
        truths: Parameter sets of ``law``, in units normalized by ``reference``.
        reference: The model and dataset size the truths' parameters are normalized by.
        cost: The cost law.
        target_budgets: Budgets at which the optimal model size is scored.
        law: The law the truths are of, and the one fit.
        run_sd: Run-to-run noise.
        eval_sd: Evaluation noise.
        evaluations_per_run: Evaluations per simulated run.
        replications: Simulated experiments per truth.
        n_draws: Bootstrap draws per fit.
        seed: Seed for the simulations.
        predictors: Names of the model-size and dataset-size columns.

    Returns:
        One row per truth, method (``"law"`` or ``"isoflop"``) and target budget: ``bias`` and ``rmse`` of
        ``log10 N*``, ``coverage`` of its 95% interval, the median interval ``width`` in decades,
        ``p_failed`` (fits that produced no estimate), and the design's ``total_cost`` and
        ``log_correlation`` between the predictors.

    Examples:
        >>> from simple_scaling_laws.cost import cost_model
        >>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
        >>> plan = isoflop_grid([1e15, 1e16, 1e17], chinchilla, centre=20.0)
        >>> scores = score_design(
        ...     plan,
        ...     truths=[{"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}],
        ...     reference=(1e7, 1e8),
        ...     cost=chinchilla,
        ...     target_budgets=[1e19],
        ...     replications=3,
        ...     n_draws=60,
        ... )
        >>> scores.select("method", "target_budget", "p_failed").rows()
        [('law', 1e+19, 0.0), ('isoflop', 1e+19, 0.0)]
    """
    from .api import fit
    from .isoflop import IsoflopError, fit_isoflop
    from .laws import build_law
    from .simulate import simulate_runs

    configurations = list(zip(design[predictors[0]].to_list(), design[predictors[1]].to_list(), strict=True))
    runs_per_point = design["runs"].to_list() if "runs" in design.columns else [1] * len(configurations)
    expanded = [c for c, r in zip(configurations, runs_per_point, strict=True) for _ in range(int(r))]
    reference_array = np.asarray(reference, dtype=float)
    instance = build_law(law, (predictors[0],), (predictors[1],))
    total_cost = float(sum(cost(n, d) for n, d in expanded))
    correlation = design_correlation(design, predictors) if "runs" in design.columns else float("nan")
    rows = []
    for t, truth in enumerate(truths):
        true_optima = [_true_log10_optimum(truth, instance, reference_array, cost, b) for b in target_budgets]
        results: dict[str, list[tuple[float, float, float] | None]] = {"law": [], "isoflop": []}
        for r in range(replications):
            frame = simulate_runs(
                {"test_loss__ce": truth},
                law=law,
                configurations=expanded,
                runs_per_config=1,
                evaluations_per_run=evaluations_per_run,
                run_sd=run_sd,
                eval_sd=eval_sd,
                predictors=predictors,
                reference=reference_array,
                seed=seed + 1000 * t + r,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frontier = fit(frame, law=law, n_draws=n_draws, seed=r).optimal_allocation(
                    target_budgets, cost
                )
                results["law"].append(
                    [
                        tuple(
                            np.log10(frontier[f"{predictors[0]}__{q}"][i]) for q in ("median", "q025", "q975")
                        )
                        for i in range(len(target_budgets))
                    ]
                )
                try:
                    predicted = fit_isoflop(frame, cost=cost, n_draws=n_draws, seed=r).predict(target_budgets)
                    results["isoflop"].append(
                        [
                            tuple(
                                np.log10(predicted[f"{predictors[0]}__{q}"][i])
                                for q in ("median", "q025", "q975")
                            )
                            for i in range(len(target_budgets))
                        ]
                    )
                except (IsoflopError, PredictionError, TypeError):
                    results["isoflop"].append(None)
        for method, outcomes in results.items():
            for i, budget in enumerate(target_budgets):
                usable = [o[i] for o in outcomes if o is not None and all(np.isfinite(o[i]))]
                errors = np.array([u[0] - true_optima[i] for u in usable])
                rows.append(
                    {
                        "truth": t,
                        "method": method,
                        "target_budget": float(budget),
                        "bias": float(errors.mean()) if usable else None,
                        "rmse": float(np.sqrt((errors**2).mean())) if usable else None,
                        "coverage": float(np.mean([u[1] <= true_optima[i] <= u[2] for u in usable]))
                        if usable
                        else None,
                        "width": float(np.median([u[2] - u[1] for u in usable])) if usable else None,
                        "p_failed": 1.0 - len(usable) / replications,
                        "total_cost": total_cost,
                        "log_correlation": correlation,
                    }
                )
    return pl.DataFrame(rows)
