# simple-scaling-laws

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/simple-scaling-laws)](https://pypi.org/project/simple-scaling-laws/)
[![Tests](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/gh/mmcdermott/simple-scaling-laws/branch/main/graph/badge.svg)](https://codecov.io/gh/mmcdermott/simple-scaling-laws)
[![Code Quality](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/code-quality-main.yaml)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/simple-scaling-laws#license)

> **Warning: internal research tooling, largely AI-generated.** This package was written primarily by an AI
> coding agent to support internal experiments in the McDermott Health AI Lab. It is public so it can
> be installed and cited easily, not because it is a hardened, externally supported library. Read the
> statistics section before you trust a number from it, check the warnings it emits, and treat its
> extrapolations as triage signal rather than evidence.

Fit empirical scaling laws to repeated ML training/evaluation records, with conservative uncertainty,
and predict performance at model or dataset sizes you have not run yet.

It exists to answer one question for an automated experiment platform: *given a handful of models
trained at different scales, is this hypothesis worth pursuing at a larger scale?* Answering it well
requires being honest about how little a small experiment tells you, so the package is opinionated
about the statistics and quiet about the knobs.

## Install

```bash
uv add simple-scaling-laws # or: uv pip install simple-scaling-laws
```

## Quick start

```bash
scaling-laws fit runs.parquet --law separable-power --output experiment.slaw

scaling-laws predict experiment.slaw points.parquet --output predictions.parquet
```

The Python API is the same thing, and is a first-class interface rather than a wrapper:

```pycon
>>> from simple_scaling_laws import fit
>>> from simple_scaling_laws.simulate import simulate_runs
>>> runs = simulate_runs(
...     {
...         "test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.30, "B": 1.5, "beta": 0.25},
...         "test_metric__auroc": {"E": 0.92, "A": -0.15, "alpha": 0.40, "B": -0.08, "beta": 0.30},
...     },
...     runs_per_config=2,
...     evaluations_per_run=5,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
... )
>>> model = fit(runs, law="separable-power", n_draws=200, seed=0)
>>> model.targets
('test_loss__cross_entropy', 'test_metric__auroc')

```

`simulate_runs` generated that table from a known law, so we can check the fit recovers it:

```pycon
>>> {k: round(v, 2) for k, v in model.params("test_loss__cross_entropy").items()}
{'E': 1.04, 'A': 1.96, 'alpha': 0.3, 'B': 1.5, 'beta': 0.25}

```

Prediction returns a dataframe, so plotting is whatever you already use:

```pycon
>>> points = pl.DataFrame(
...     {
...         "point_id": ["current", "10x-model", "10x-data"],
...         "model_size__n_params": [1e8, 1e9, 1e8],
...         "dataset_size__n_subjects": [1e5, 1e5, 1e6],
...     }
... )
>>> predictions = model.predict(points, quantiles=[0.025, 0.5, 0.975])
>>> for row in predictions.iter_rows(named=True):
...     print(
...         f"{row['point_id']:<10} {row['domain']:<14} "
...         f"{row['test_loss__cross_entropy__median']:.3f} "
...         f"[{row['test_loss__cross_entropy__q025']:.3f}, "
...         f"{row['test_loss__cross_entropy__q975']:.3f}]"
...     )
current    interpolation  2.856 [2.838, 2.878]
10x-model  extrapolation  2.367 [2.323, 2.413]
10x-data   extrapolation  2.487 [2.456, 2.530]

```

## Input format

One table, **long over evaluation resamples and wide over metrics**. Column roles are inferred from
prefixes, so a conventionally named table needs no configuration at all:

| column            | meaning                                                   |
| ----------------- | --------------------------------------------------------- |
| `training_run_id` | one independently trained model (**required**)            |
| `train_set_id`    | the training split used                                   |
| `test_set_id`     | one evaluation resample; **reuse it across models**       |
| `optimizer_seed`  | the seed that produced this model                         |
| `model_size__*`   | model-scale predictors, e.g. `model_size__n_params`       |
| `dataset_size__*` | dataset-scale predictors, e.g. `dataset_size__n_subjects` |
| `test_loss__*`    | test losses; the primary scaling target                   |
| `train_loss__*`   | train losses                                              |
| `test_metric__*`  | test metrics, e.g. `test_metric__auroc`                   |
| `train_metric__*` | train metrics                                             |

Every target column found is fit, each with its own parameters and its own uncertainty. Columns that
match nothing are ignored, so extra bookkeeping is harmless.

Three structural rules carry all the statistical weight:

1. **One `training_run_id` per independently trained model.** Different seeds or different splits
    mean different runs.
2. **Many rows per run, one per evaluation resample.** The spread within a run estimates
    evaluation noise. Do not pre-aggregate into bootstrap quantiles -- give the package the rows.
3. **Reuse `test_set_id` across models.** When the same resample scores several models, the package
    detects the shared noise instead of mistaking it for a real difference between them.

Non-conventional column names can be mapped without renaming anything:

```pycon
>>> renamed = runs.rename({"training_run_id": "run", "test_set_id": "bootstrap_id"})
>>> remapped = fit(
...     renamed,
...     columns={"training_run_id": "run", "test_set_id": "bootstrap_id"},
...     n_draws=50,
...     seed=0,
... )
>>> remapped.primary_target
'test_loss__cross_entropy'

```

or in a small YAML config passed to `--config`:

```yaml
law: separable-power

columns:
  training_run_id: run
  test_set_id: bootstrap_id
```

## What it fits

Seven law families. Each is linear in its offset and amplitudes, which are solved exactly, and
nonlinear in a handful of other parameters -- exponents, and for the richer laws a shift, a ratio or
a break -- which are searched:

```pycon
>>> from simple_scaling_laws import available_laws
>>> for name, description in available_laws().items():
...     print(f"{name}\n    {description}")
separable-power
    E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)
multiplicative-power
    E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)
shifted-separable-power
    E + sum_i A_i * (x_i + s_i)**-alpha_i (power terms with a finite small-scale limit)
kaplan-joint-power
    E + A * (N**-rho + R / D)**beta (Kaplan et al. joint model/data law)
skaling-joint-power
    E + A * (N**-alpha + R * D**-beta)**k (Videau et al. Skaling law: coupled model/data terms)
bounded-rational-power
    (E + I * g) / (1 + g), g = sum_i A_i * x_i**-alpha_i (plateau I at small scale, floor E at large)
broken-power
    E + A * x**-c0 * (1 + (x / d)**(1 / f))**(-c1 * f) (one smooth change of slope; single predictor)

```

With one model-size and one dataset-size column, `separable-power` is the conventional
`E + A·N^-α + B·D^-β`. Start there. The other families each relax one specific assumption it makes,
at the price of more parameters and a larger grid, and each except `multiplicative-power` (a single
joint power term, `E + A·N^-α·D^-β`) has a section below saying when it is worth that price. A lower in-sample residual is not by itself a reason to prefer one: every richer
law here can fit separable data at least as well as the separable law does.

Losses are fit with non-negative amplitudes, which gives the usual "irreducible floor plus decaying
terms" reading. Metrics get free-signed amplitudes so an increasing metric such as AUROC can approach
its asymptote from below:

```pycon
>>> auroc = model.params("test_metric__auroc")
>>> auroc["A"] < 0 and auroc["B"] < 0
True

```

### The examples below

The richer laws are shown on a 4x4 grid, and some are compared against data that follow the plain
separable law exactly:

```pycon
>>> grid = {"model_sizes": (1e6, 1e7, 1e8, 1e9), "dataset_sizes": (1e3, 1e4, 1e5, 1e6)}
>>> separable = simulate_runs(
...     {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}},
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
...     **grid,
... )

```

There is deliberately no law here in which a larger model does *worse* when data are scarce. The
obvious form -- the separable law minus a joint term `C·N^-γ·D^-δ` -- was tried and removed. Any
`C` large enough to couple the two resources also makes the loss rise with model size somewhere:
always at small enough data, and at every data size when `γ < α`. So where the curve turns up is
set by the formula, not by the data. Videau et al. (2026) reject the form on the same grounds. The
two joint laws below couple model size and data while staying monotone in both. For a curve that
genuinely turns, use `broken-power` along one axis.

### When the small end is not yet a power law

A power law is a straight line on log-log axes all the way down, and diverges at zero scale. Real
curves flatten toward something finite at the small end of an experiment, and those pre-asymptotic
points drag a plain power law's exponent toward zero. `shifted-separable-power` puts a non-negative
shift inside each power term, `E + A·(N + s_N)^-α + B·(D + s_D)^-β`, which gives the term a finite
value at zero scale and leaves its large-scale behavior alone. Its single-predictor form is the
saturating law `a·(C + d)^-b + c` of Zhai et al. (2022). It is not the "M3" form of Bansal et al.
(2022), `α·(D^-1 + C)^p`, which shifts the reciprocal of the scale and bounds the large end instead.

The cost of ignoring a shift that is there is not subtle. Fitting the plain law to shifted data:

```pycon
>>> truth = {"E": 1.0, "A": 2.0, "B": 1.5, "alpha": 0.5, "beta": 0.4, "s_n_params": 0.3, "s_n_subjects": 0.2}
>>> bent = simulate_runs(
...     {"test_loss__cross_entropy": truth},
...     law="shifted-separable-power",
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
...     **grid,
... )
>>> plain = fit(bent, law="separable-power", n_draws=200, seed=0)
>>> shifted = fit(bent, law="shifted-separable-power", n_draws=200, seed=0)
>>> {k: round(v, 2) for k, v in plain.params("test_loss__cross_entropy").items()}
{'E': -15.81, 'A': 12.84, 'alpha': 0.04, 'B': 7.21, 'beta': 0.05}
>>> [note.code for note in plain.warnings]
['lack_of_fit']
>>> {k: round(v, 2) for k, v in shifted.params("test_loss__cross_entropy").items()}
{'E': 1.0, 'A': 1.99, 'B': 1.51, 'alpha': 0.5, 'beta': 0.39, 's_n_params': 0.3, 's_n_subjects': 0.19}

```

Two decades beyond the grid, where the true loss is 1.10, the plain law predicts a negative one and
its interval does not reach the truth; the shifted law's does:

```pycon
>>> far = {"model_size__n_params": [1e11], "dataset_size__n_subjects": [1e8]}
>>> for name, candidate in (("plain", plain), ("shifted", shifted)):
...     row = candidate.predict(far).row(0, named=True)
...     print(
...         f"{name:<8} {row['test_loss__cross_entropy__median']:6.2f} "
...         f"[{row['test_loss__cross_entropy__q025']:.2f}, {row['test_loss__cross_entropy__q975']:.2f}]"
...     )
plain     -1.33 [-2.18, -0.27]
shifted    1.10 [1.05, 1.15]

```

`lack_of_fit` on a separable fit is therefore the cue to try this law. It needs **four** scales of
every predictor rather than three -- with three there are two contrasts from which to learn an
amplitude, an exponent and a shift -- and says so with an `error`-severity
`too_few_predictor_levels` note if it does not get them. On data that are already a clean power law
the shifts sit at zero and are reported as `parameter_at_bound`. Measured on 4x4 and 5x5 designs, its
nominal 95% prediction intervals covered the truth 88-98% of the time, inside the grid and a decade
beyond it.

### When one resource is the bottleneck

`kaplan-joint-power` is the joint law of Kaplan et al. (2020), `E + A·(N^-ρ + R/D)^β`, with an
irreducible floor `E` added. Whichever resource is scarcer sets the loss: with ample data it is a
power law in model size (exponent `ρ·β`), with an ample model it is a power law in data (exponent
`β`), and in between, adding more of the abundant resource does almost nothing.

```pycon
>>> truth = {"E": 1.0, "A": 2.0, "rho": 0.5, "beta": 0.6, "R": 1.0}
>>> limited = simulate_runs(
...     {"test_loss__cross_entropy": truth},
...     law="kaplan-joint-power",
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
...     **grid,
... )
>>> kaplan = fit(limited, law="kaplan-joint-power", n_draws=200, seed=0)
>>> {k: round(v, 2) for k, v in kaplan.params("test_loss__cross_entropy").items()}
{'E': 1.01, 'A': 2.0, 'rho': 0.5, 'beta': 0.6, 'R': 1.0}
>>> sweep = kaplan.predict(
...     {
...         "model_size__n_params": [1e6, 1e7, 1e8, 1e9, 1e10, 1e11] * 2,
...         "dataset_size__n_subjects": [1e3] * 6 + [1e6] * 6,
...     }
... )
>>> losses = [round(v, 2) for v in sweep["test_loss__cross_entropy__median"]]
>>> losses[:6]  # scarce data: a hundred thousand times the model buys almost nothing
[18.53, 17.42, 17.06, 16.94, 16.91, 16.89]
>>> losses[6:]  # ample data: the model is the bottleneck, and scaling it pays
[6.65, 3.86, 2.47, 1.79, 1.47, 1.34]

```

It is monotone in both resources, so it describes saturation, not a larger model being *worse*. It is defined for exactly one model-size and one dataset-size
column, and both must actually vary.

Measured on 4x4 designs, the Kaplan law's nominal 95% prediction intervals covered the truth
92-100% of the time, inside the grid and a decade beyond it. On a 3x3 grid with two runs per
configuration the Kaplan law's covered 93-95%, but only because the bootstrap puts back noise that
a fit this close to saturated absorbs (78-85% without it); prefer the larger grid.

### When model size and data are coupled, but neither is ever harmful

`skaling-joint-power` is the Skaling law of Videau, Youbi-Idrissi, Lopez-Paz & Ahuja (2026),
`E + A·(N^-α + R·D^-β)^k`: the two power terms of the separable law, summed and raised to one
*coupling exponent* `k`. At `k = 1` it is exactly `separable-power`; with the inner data exponent fixed
at one it is `kaplan-joint-power`. For `0 < k < 1` each resource makes the other more valuable --
the coupling the authors measured on large language-model grids, with `k` from about 0.3 to 0.9 --
and the loss still falls in both. With ample data it is a power law in model size with exponent
`α·k`, with an ample model a power law in data with exponent `β·k`.

```pycon
>>> truth = {"E": 1.0, "A": 2.0, "alpha": 0.5, "beta": 0.4, "k": 0.5, "R": 1.0}
>>> joint = simulate_runs(
...     {"test_loss__cross_entropy": truth},
...     law="skaling-joint-power",
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
...     **grid,
... )
>>> skaling = fit(joint, law="skaling-joint-power", n_draws=200, seed=0)
>>> {k: round(v, 2) for k, v in skaling.params("test_loss__cross_entropy").items()}
{'E': 1.03, 'A': 1.98, 'alpha': 0.5, 'beta': 0.4, 'k': 0.5, 'R': 0.99}

```

On the separable data from the section above, which have no coupling at all, the fit puts `k` at
one and says nothing, because the separable law it then reduces to is the right answer. Its `A` and
`R·A` are then that law's two amplitudes:

```pycon
>>> skaling_on_plain = fit(separable, law="skaling-joint-power", n_draws=200, seed=0)
>>> round(skaling_on_plain.params("test_loss__cross_entropy")["k"], 2)
1.0
>>> [note.code for note in skaling_on_plain.warnings]
[]

```

It is defined for exactly one model-size and one dataset-size column. The paper fit 117 to 404 runs
per grid; on simulated 3x3 grids with two runs per configuration `k` still came back within about
0.01 of the truth and nominal 95% prediction intervals covered 88-100%.

### When the experiment reaches down to chance

Every law above grows without limit as scale shrinks. A real loss does not: a model trained on
almost nothing sits at chance and stays there. `bounded-rational-power` has a plateau `I` at the
small end as well as a floor `E` at the large one, and moves between them as a weighted average,
`(E + I·g) / (1 + g)`, whose weight `g = A·N^-α + B·D^-β` is an ordinary separable sum of power
terms. It is the M4 estimator of Alabdulmohsin, Neyshabur & Zhai (2022) with its shape exponent
(the paper's α, in `(f - E) / (I - f)^α = g`) fixed at one -- the
one case that can be written down in closed form, and so the one whose two limits the fitter can
solve exactly rather than search for.

```pycon
>>> truth = {"E": 1.0, "I": 4.0, "A": 1.0, "B": 0.8, "alpha": 0.5, "beta": 0.4}
>>> saturating = simulate_runs(
...     {"test_loss__cross_entropy": truth},
...     law="bounded-rational-power",
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
...     **grid,
... )
>>> bounded = fit(saturating, law="bounded-rational-power", n_draws=200, seed=0)
>>> {k: round(v, 2) for k, v in bounded.params("test_loss__cross_entropy").items()}
{'E': 1.03, 'I': 4.0, 'A': 1.01, 'B': 0.78, 'alpha': 0.49, 'beta': 0.41}

```

What it buys is a sane answer below the grid. Three decades under the smallest configuration the
true loss is 3.99, just short of the plateau. The separable law, fit to the same runs, has nothing
to stop it:

```pycon
>>> tiny = {"model_size__n_params": [1e3], "dataset_size__n_subjects": [1e0]}
>>> for name, candidate in (("bounded", bounded), ("separable", fit(saturating, n_draws=200, seed=0))):
...     row = candidate.predict(tiny).row(0, named=True)
...     print(
...         f"{name:<10} {row['test_loss__cross_entropy__median']:5.2f} "
...         f"[{row['test_loss__cross_entropy__q025']:.2f}, {row['test_loss__cross_entropy__q975']:.2f}]"
...     )
bounded     3.99 [3.95, 4.03]
separable   8.14 [5.06, 144.22]

```

The plateau is only identified if the experiment reaches far enough down to see the curve bend
toward it. On data that are still a clean power law `I` runs off to its bound while the weights
shrink to compensate, and the fit is reported as `ill_conditioned_fit`: use `separable-power`. Both
limits are estimated, never assumed; for a metric confined to `[0, 1]` they are limits on the logit
scale the metric is fit on. Its intervals covered the truth 82-98% of the time on 4x4 designs and
97-98% on a 3x3 grid with two runs per configuration -- there only because the bootstrap puts back
noise the fit absorbed (75-82% without it), so this law too wants the larger grid.

On the plain separable data above, the note names the parameters that trade off against each other
with almost no change in the fitted curve -- read it as the instruction to drop back to
`separable-power`:

```pycon
>>> unbent = fit(separable, law="bounded-rational-power", n_draws=200, seed=0)
>>> next(note for note in unbent.warnings if note.code == "ill_conditioned_fit").details["parameters"]
['I', 'A', 'B']

```

### When the slope changes partway

Every law above has one slope on log-log axes. `broken-power` has two, joined smoothly at a break
scale `d`: `E + A·x^-c0·(1 + (x/d)^(1/f))^(-c1·f)`, the one-break Broken Neural Scaling Law of
Caballero et al. (2022). Below the break the excess loss falls as `x^-c0`, above it as
`x^-(c0+c1)`, and `c1` may have either sign -- so the curve can steepen, flatten, or, when
`c1 < -c0`, turn back up. It is a law for a sweep along **one** predictor:

```pycon
>>> truth = {"E": 1.0, "A": 2.0, "c0": 0.3, "c1": -0.6, "d": 3.0, "f": 0.8}
>>> sizes = tuple(10.0**k for k in range(6, 14))
>>> sweep = simulate_runs(
...     {"test_loss__cross_entropy": truth},
...     model_sizes=sizes,
...     dataset_sizes=(),
...     law="broken-power",
...     runs_per_config=2,
...     evaluations_per_run=4,
...     run_sd=0.01,
...     eval_sd=0.02,
...     seed=0,
... )
>>> broken = fit(sweep, law="broken-power", n_draws=200, seed=0)
>>> {k: round(v, 2) for k, v in broken.params("test_loss__cross_entropy").items()}
{'E': 0.99, 'A': 2.0, 'c0': 0.3, 'c1': -0.6, 'd': 2.98, 'f': 0.8}
>>> at_sizes = {"model_size__n_params": list(sizes)}
>>> [round(v, 1) for v in broken.predict(at_sizes)["test_loss__cross_entropy__median"]]
[23.4, 12.2, 6.6, 3.9, 3.0, 4.0, 6.8, 12.6]

```

A single power law cannot turn, so on the same runs it settles for a floor through the middle of
them, and says the fit is poor:

```pycon
>>> single = fit(sweep, n_draws=200, seed=0)
>>> [round(v, 1) for v in single.predict(at_sizes)["test_loss__cross_entropy__median"]]
[23.3, 11.8, 8.6, 7.3, 6.8, 6.2, 6.2, 6.2]
>>> "lack_of_fit" in [note.code for note in single.warnings]
True

```

Three things to know before using it:

- **It needs at least six scales of its predictor** -- it has six parameters on one axis -- and
    raises `too_few_predictor_levels` otherwise.
- **The break can only be found inside the range that was run.** A break estimated at either end of
    it is reported as `parameter_at_bound`, and means no break was observed. An extrapolation simply
    continues the upper slope; it cannot anticipate another change of regime beyond the grid.
- **Its intervals rely on knowing the noise.** With six parameters on eight scales the fit passes
    almost exactly through some of them, and the bootstrap puts back the noise the fit absorbed there
    from the separately estimated run and evaluation variances. With that, nominal 95% prediction
    intervals covered 90-97% in simulation; without it they covered 70-88%. Replicate runs at a few
    scales make that noise estimate direct rather than inferred from the residuals.

## Bounded metrics

AUROC lives in `[0, 1]`; a fitted asymptote of 4.9 is not a statement about a classifier. Metrics
known to be confined to the unit interval are therefore fit on their **logit** and mapped back for
prediction, which makes the bound hold by construction rather than by clipping afterwards. It also
makes the constant-variance assumption far more plausible, since an AUROC near 0.99 has much less
room to move than one near 0.7.

The common metrics are recognized by name, with no configuration:

```pycon
>>> from simple_scaling_laws.metrics import describe
>>> for name in ("auroc", "auprc", "accuracy", "brier", "ece", "cross_entropy"):
...     kind = describe(name, is_loss=name == "cross_entropy")
...     print(f"{name:<14} {kind.support:<5} {kind.direction} is better")
auroc          unit  higher is better
auprc          unit  higher is better
accuracy       unit  higher is better
brier          unit  lower is better
ece            unit  lower is better
cross_entropy  real  lower is better

```

Direction matters because the role prefix does not carry it: `test_metric__auroc` improves upward
and `test_metric__brier` improves downward, and only the direction turns a difference between two
systems into a win rate. Anything the table does not recognize is left untransformed, with its
direction marked as assumed rather than known. Describe such a target explicitly:

```pycon
>>> described = fit(
...     runs,
...     targets={"test_metric__auroc": {"support": "unit", "direction": "higher"}},
...     n_draws=50,
...     seed=0,
... )
>>> described.target_kind("test_metric__auroc").support
'unit'

```

The same goes in a config file under a `targets:` key. Note that a transformed target's *parameters*
are on the logit scale — its `E` is an asymptote in logit units — while everything `predict` returns
is on the metric's own scale, and cannot leave it:

```pycon
>>> auroc = described.predict(
...     {
...         "model_size__n_params": [1e6, 1e20],
...         "dataset_size__n_subjects": [1e3, 1e20],
...     }
... )
>>> lo, hi = auroc["test_metric__auroc__q025"], auroc["test_metric__auroc__q975"]
>>> bool((lo >= 0).all() and (hi <= 1).all())
True

```

## Comparing systems

Fitting a law per system says how each one scales. The question a triage platform actually asks is
comparative — *of these arms, which is worth scaling up?* — and two overlapping intervals are
perfectly compatible with one arm being reliably better. What you need is the distribution of the
**difference**:

```bash
scaling-laws compare points.parquet \
	--model baseline=baseline.slaw \
	--model tuned=tuned.slaw \
	--reference baseline
```

```pycon
>>> from simple_scaling_laws import compare
>>> def arm(alpha, seed):
...     return simulate_runs(
...         {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": alpha, "B": 1.5, "beta": 0.25}},
...         runs_per_config=2,
...         evaluations_per_run=4,
...         run_sd=0.01,
...         eval_sd=0.02,
...         seed=seed,
...     )
>>> arms = {
...     "baseline": fit(arm(0.30, 0), n_draws=300, seed=0),
...     "tuned": fit(arm(0.34, 1), n_draws=300, seed=0),
...     "risky": fit(arm(0.28, 2), n_draws=300, seed=0),
... }
>>> table = compare(
...     arms,
...     {
...         "point_id": ["now", "10x"],
...         "model_size__n_params": [1e8, 1e9],
...         "dataset_size__n_subjects": [1e5, 1e5],
...     },
...     reference="baseline",
... )
>>> for row in table.iter_rows(named=True):
...     print(
...         f"{row['point_id']:<4} {row['system']:<9} "
...         f"{row['value__median']:.3f}  best {row['p_best']:>4.0%}  "
...         f"better than baseline {row['p_better_than_reference']:>4.0%}"
...     )
10x  baseline  2.352  best   0%  better than baseline   0%
10x  risky     2.382  best   0%  better than baseline  29%
10x  tuned     2.244  best 100%  better than baseline 100%
now  baseline  2.848  best   0%  better than baseline   0%
now  risky     2.891  best   0%  better than baseline   3%
now  tuned     2.747  best 100%  better than baseline 100%

```

Any number of arms works, and the output is long — one row per point per arm — so ten arms do not
become a hundred columns. `p_best` is the probability that arm is the best of the set at that point;
with a `reference`, each arm also gets the distribution of its difference from the reference
(`difference__median`, `difference__q025`, …) and the probability it beats it. Direction is handled
for you: a loss is better when smaller, AUROC when larger.

### Why the draws have to be paired

Arms run at the same scales share a configuration-level deviation — a split that is hard for one is
hard for the other, and where the law misfits it bends both curves the same way. That shared part
cancels in the difference, so the difference is pinned down far more sharply than either level is.

The wild bootstrap makes this nearly free. It perturbs residuals with a multiplier drawn once per
configuration; if every arm gets the *same* multiplier sequence, the perturbation of the difference
is driven by the difference's own residual. That is right whether or not the arms share splits: when
their residuals are correlated the difference is sharp, and when they are independent the arithmetic
recovers the independent answer on its own. This is common random numbers, the standard device for
comparing simulations, and it is why a paired t-test beats an unpaired one.

It matters a great deal. Measured against the true sampling spread of the difference, over 120
simulated worlds:

| shared configuration sd | true sd(A−B) | paired draws   | independent draws |
| ----------------------- | ------------ | -------------- | ----------------- |
| 0.00                    | 0.0329       | 0.0413 (1.26x) | 0.0430 (1.31x)    |
| 0.05                    | 0.0330       | 0.0416 (1.26x) | 0.1330 (4.03x)    |
| 0.15                    | 0.0338       | 0.0442 (1.31x) | 0.3715 (11.00x)   |

Differencing two independently-seeded fits is **4x to 11x too wide** once the arms share structure —
wide enough that no real difference is ever detectable. Fits pair automatically when they were run
with the same `seed` on the same configuration grid, which is the normal case for arms of one
experiment; `compare` verifies it, reports `paired`, and warns loudly when it cannot.

## The statistics, briefly

**Evaluation rows are not training evidence.** Each run is reduced to one observation -- its mean --
plus a count of how many evaluations that mean averages. Scoring one model on a hundred bootstrap
resamples makes that model's mean more precise; it does not make it a hundred models.

**Two variance components.** Spread *within* a run estimates finite-evaluation-set noise; spread
*between* independently trained models at the same scale estimates training-run stochasticity. The
run-level variance is pooled across configurations rather than estimated separately from the two or
three repeats you have at each. Without replicate runs it falls back to the scatter around the fitted
curve, which also contains the law's misspecification -- biased upward, which is the safe direction.

**Shared test sets are recognized.** When models share `test_set_id` values, the correlation between
their evaluation residuals is estimated and reported, and only the part of evaluation noise that is
*not* shared is treated as averaging away when runs are compared. This matters more than it sounds:
with nine tenths of the evaluation noise coming from the resamples themselves, subtracting the
*marginal* within-run variance drives the estimated run-to-run variance to exactly zero in about half
of simulated experiments -- which would make a `new-run` prediction no wider than a `mean` one.
Subtracting only the unshared part recovers the true value.

**Uncertainty is a wild cluster bootstrap over configurations.** Residuals around the fitted curve
are corrected and re-signed with Webb's six-point weights, drawn once per *configuration* so that a
scale's shared lack of fit is preserved rather than averaged away. This was chosen because it is
assumption-free about the error distribution, stays valid with the four-to-a-dozen configurations a
real scaling experiment has, keeps the `(N, D)` design fixed instead of resampling it into
degeneracy, and is conservative: residuals carry the law's own misspecification into the intervals.
Draws -- not quantiles -- are stored, so any summary can be computed later.

**The intervals are calibrated against simulation, not asserted.** Resampling residuals naively
produces intervals that are badly too narrow at these sample sizes: generating from a known law and
counting how often the nominal 95% intervals contained it gave **74%** at nine configurations and
**87%** at sixteen. Two corrections fix it -- a degrees-of-freedom factor for the parameters the fit
consumed at the level the bootstrap resamples at, and a Student's-t rather than normal reference for
the interval endpoints -- and take measured coverage to **91-95%** across the designs this package
targets. A third correction matters when a law has nearly as many parameters as configurations: the
fit then passes almost exactly through some of them, their residuals understate the noise, and each
configuration's pseudo-residual is floored at the share of its (separately estimated) noise that the
fit absorbed. It never shrinks a configuration's pseudo-residual, so it pushes intervals wider, not
narrower. `tests/test_coverage.py` re-measures this on every CI run, so it cannot silently regress.

That calibration was measured on `separable-power`. Every other law is held to the same test
(`tests/test_law_families.py`), and its section above reports what it measured.

**Correlations are computed across runs.** Loss/metric correlations use run-level means, with
intervals from resampling configurations. The naive evaluation-row correlation is reported alongside,
with its row count, purely so the inflated sample size is visible -- its estimate can land close to the
run-level one while claiming five times the evidence:

```pycon
>>> correlations = model.metric_correlations()
>>> for row in correlations.iter_rows(named=True):
...     print(
...         f"{row['target']}: run-level r={row['pearson']:.3f} over {row['n_runs']} runs "
...         f"(evaluation-row r={row['evaluation_row_pearson']:.3f} over "
...         f"{row['n_evaluation_rows']} rows)"
...     )
test_metric__auroc: run-level r=-0.979 over 18 runs (evaluation-row r=-0.971 over 90 rows)

```

## Allocating a budget

A fitted law says what loss a configuration reaches. Splitting a budget between model size and data
needs a second model, of what each configuration *costs*. Cost laws come from a registry of named
forms, just like scaling laws, each over one model-size predictor `N` and one dataset-size predictor
`D`:

```pycon
>>> from simple_scaling_laws import available_cost_forms, cost_model, fit_cost_model
>>> for name, description in available_cost_forms().items():
...     print(f"{name}\n    {description}")
power
    k * N**a * D**b (e.g. k=6, a=b=1: training FLOPs of a transformer)
per-example
    D * (k * N**a + c) (per-example cost growing with N, plus a fixed overhead)

```

A cost model is either declared outright, or fitted to runs that recorded what they cost -- a
`compute__*` column, one value per run -- with any parameters you already know held fixed:

```pycon
>>> chinchilla = cost_model("power", k=6.0, a=1.0, b=1.0)
>>> costed = runs.with_columns(
...     (6.0 * pl.col("model_size__n_params") * pl.col("dataset_size__n_subjects")).alias("compute__flops")
... )
>>> fitted_cost = fit_cost_model(costed, form="power", fixed={"b": 1.0})
>>> fitted_cost.description, fitted_cost.fitted
('power: k=6, a=1, b=1', ('k', 'a'))

```

Measured cost is better than a formula wherever you have it. `D` must be what each run *processed*
-- tokens or examples seen, every epoch included -- and a run that stops early processes less than its
schedule said. Logged compute captures that; a formula written in advance does not. In PyTorch,
`torch.utils.flop_counter.FlopCounterMode` counts a forward and backward pass; multiplied by the
examples processed, that is the run's cost. Counting one model is not enough to *declare* a cost law,
because how cost grows with `N` depends on how the model is scaled up (wider or deeper, longer context,
embeddings that cost nothing per example): record it for several sizes and fit.

Given a cost model, `optimal_allocation` minimizes every bootstrap draw of the law along the curve of
equal cost, so the optimal model size, dataset size and loss come with intervals:

```pycon
>>> frontier = model.optimal_allocation([1e11, 1e13], chinchilla)
>>> for row in frontier.iter_rows(named=True):
...     print(
...         f"{row['budget']:.0e}: N* = {row['model_size__n_params__median']:.2g} "
...         f"[{row['model_size__n_params__q025']:.2g}, {row['model_size__n_params__q975']:.2g}], "
...         f"loss {row['test_loss__cross_entropy__median']:.3f}"
...     )
1e+11: N* = 1e+07 [9.8e+06, 1.1e+07], loss 5.346
1e+13: N* = 8.2e+07 [7.7e+07, 8.8e+07], loss 3.330
>>> exponents = model.allocation_exponents([1e11, 1e13], chinchilla)
>>> round(exponents["model_size"]["median"], 2)  # N* grows as C**0.45 here
0.45

```

Simulated 100 times at a hundred times the largest budget run, the 95% interval for `N*` covered the
true optimum 97-98% of the time when the law was right. Two warnings matter here:

- **`no_interior_optimum`**: a draw keeps improving all the way to the edge of the search, three
    decades past the observed range. The law sets no limit on how much to spend on one resource, so
    the optimum is not determined. `multiplicative-power`, monotone along any iso-cost curve, always
    does this.
- **`optimum_outside_observed_domain`**: the optimum is an extrapolation of the law's *shape*, even
    when the budget itself was run. That is where a law that is only slightly wrong goes most wrong.

The same is available from the command line:

```bash
scaling-laws allocate experiment.slaw --budget 1e18 --budget 1e19 --cost power:k=6,a=1,b=1
scaling-laws allocate experiment.slaw --budget 1e18 --cost power:b=1 --cost-records runs.parquet
```

### Checking the frontier without the law: isoFLOP profiles

The frontier above is only as good as the law's shape out at the optimum. The law-free check is
Chinchilla's "Approach 2": train several model sizes at each of a few fixed budgets, find the bottom of
each budget's valley, and fit a line of `log N*` against `log C`. It needs a different experiment --
sweeps along curves of equal cost -- which `simulate_runs` can lay out from explicit configurations:

```pycon
>>> import numpy as np
>>> from simple_scaling_laws import fit_isoflop
>>> sweeps = [
...     (n, float(chinchilla.dataset_size_for(budget, n)))
...     for budget in (1e15, 1e16, 1e17, 1e18)
...     for n in np.sqrt(budget / 60.0) * np.logspace(-0.6, 0.6, 6)
... ]
>>> isoflop_runs = simulate_runs(
...     {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}},
...     configurations=sweeps,
...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
...     run_sd=0.005,
...     eval_sd=0.005,
...     seed=0,
... )
>>> profiles = fit_isoflop(isoflop_runs, cost=chinchilla, n_draws=200)
>>> profiles.profiles()["bracketed"].to_list()
[True, True, True, True]
>>> round(profiles.exponent()["median"], 2)  # beta / (alpha + beta) = 0.57 for this law
0.57

```

Each profile is a cubic in `log N` by default, not Chinchilla's parabola. Along a curve of equal cost
the valley is lopsided whenever the two exponents differ, the parabola's minimum is pulled toward the
shallow side, and resampling cannot see that bias: simulated, the parabola's 95% intervals for `N*`
covered the truth only 48-72% of the time when the noise was small, the cubic's 92-100%. The cost is
one more model size per profile (five at least) and wider intervals; `degree=2` is still available.

Report the two frontiers side by side, rather than choosing between them. Where their intervals do not
overlap, at least one is wrong -- most often the law, being extrapolated -- and
`isoflop_law_disagreement` says so:

```pycon
>>> law = fit(isoflop_runs, n_draws=200, seed=0)
>>> profiles.compare(law, [1e18, 1e20])["agree"].to_list()
[True, True]

```

Profiles report `isoflop_too_few_profile_points` (fewer than five sizes, or four for a parabola),
`isoflop_minimum_not_bracketed` (the minimum lies outside the sizes run, so add sizes on the other side),
`isoflop_profile_not_convex`, `isoflop_unstable_minimum` and `isoflop_too_few_budgets` (fewer than three).
From the command line, `scaling-laws isoflop runs.parquet --cost power:k=6,a=1,b=1 --budget 1e20`.

### Planning the experiment

Before anything has run, `isoflop_grid` lays out sweeps of model size along curves of equal cost,
centred on a guess at the optimum -- a data-to-model ratio, explicit sizes, or a fit of earlier runs.
The same runs serve both frontiers: along each curve they are an isoFLOP profile, and across curves they
form a grid, rotated in `(log N, log D)`, that a joint law can be fit to.

```pycon
>>> from simple_scaling_laws import isoflop_grid, score_design, suggest_runs
>>> from simple_scaling_laws.design import design_correlation
>>> plan = isoflop_grid([1e15, 1e16, 1e17, 1e18], chinchilla, centre=20.0)
>>> plan.height, int(plan["runs"].sum()), round(design_correlation(plan), 2)
(24, 28, -0.3)

```

Every point on a curve of equal cost costs the same, so how wide each sweep reaches is free. Wider
sweeps reach the lopsided corners that pin down how model size and data trade off, and protect
against a centre guessed wrong. Scored by simulation (`score_design`, which fits both frontiers to
simulated runs of a candidate plan under plausible truths), the law's frontier kept improving with
width, and the isoFLOP profiles did best around the default of ±1.2 decades. Use `score_design` to
compare plans of your own before spending anything on them.

Once some runs exist, `suggest_runs` picks the next ones to shrink the uncertainty in the optimal model
size at the budgets you care about, per unit of compute. It treats the bootstrap draws as an approximate
posterior and asks, for each candidate point on the budgets you can afford, how much one more run there
would narrow the target, conditioning on each pick before making the next:

```pycon
>>> pilot_plan = isoflop_grid([1e15, 1e16], chinchilla, centre=20.0)
>>> pilot = simulate_runs(
...     {"test_loss__cross_entropy": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.4}},
...     configurations=list(zip(pilot_plan["model_size__n_params"], pilot_plan["dataset_size__n_tokens"])),
...     predictors=("model_size__n_params", "dataset_size__n_tokens"),
...     run_sd=0.02,
...     eval_sd=0.03,
...     seed=0,
... )
>>> next_runs = suggest_runs(
...     fit(pilot, n_draws=200, seed=0), [1e20], chinchilla, n=4, candidate_budgets=[1e17, 1e18]
... )
>>> [(f"{b:.0e}", f"{n:.2g}") for b, n in next_runs.select("budget", "model_size__n_params").rows()]
[('1e+17', '1.5e+08'), ('1e+17', '1.5e+08'), ('1e+17', '1.5e+08'), ('1e+17', '1.5e+08')]

```

All four at one point is not a bug. For a single quantity the best design has very few support points,
and here the cheaper budget buys more per unit of compute. That is efficient if the law is right and
fragile if it is not; `max_per_point=1` spreads the batch at some cost in efficiency.

Simulated 60 times, three rounds of suggested runs at two budgets -- each round refit before the next --
halved the error in `N*` a hundred times past the largest budget compared with a grid at the same
budgets, with intervals half as wide and no more compute; coverage stayed near nominal. The criterion
sees only the fitted law, so refit after every batch, and start from a grid so the isoFLOP check stays
possible. `scaling-laws plan` and `scaling-laws suggest` do the same from the command line.

## Prediction semantics

| `kind`    | answers                                                              |
| --------- | -------------------------------------------------------------------- |
| `mean`    | what does this training procedure achieve *on average* at this scale |
| `new-run` | what might *one* newly trained model achieve at this scale           |

`mean` is the default because it is the scaling law itself. `new-run` adds training-run
stochasticity, resampled from the observed run-to-run deviations rather than assumed Gaussian
when there are at least ten of them (otherwise drawn as Gaussian with the estimated run SD):

```pycon
>>> def width(frame, target="test_loss__cross_entropy"):
...     return (frame[f"{target}__q975"] - frame[f"{target}__q025"]).to_list()
>>> mean_widths = width(model.predict(points))
>>> new_run_widths = width(model.predict(points, kind="new-run"))
>>> all(n > m for n, m in zip(new_run_widths, mean_widths, strict=True))
True

```

Evaluation-set noise is deliberately not included: a scaling prediction is about the model, not about
which bootstrap resample you happen to score it on.

## Interpolation and extrapolation

Both go through `predict`. The difference is reported, not enforced -- extrapolating is the entire
point of fitting a scaling law. Every prediction carries a `domain` label and an
`extrapolation_distance`, measured in units of each predictor's observed log range, so one full
observed range beyond the edge is a distance of `1.0`:

```pycon
>>> for row in predictions.iter_rows(named=True):
...     print(f"{row['point_id']:<10} {row['domain']:<14} {row['extrapolation_distance']:.2f}")
current    interpolation  0.00
10x-model  extrapolation  0.50
10x-data   extrapolation  0.50

```

An extrapolated interval reflects *parameter* uncertainty only. It cannot tell you that the power law
stops holding two orders of magnitude out, so a `UserWarning` accompanies any extrapolated request.

## The artifact

`model.save(path)` writes a self-contained directory that loads without the original dataframe:

```pycon
>>> artifact = model.save(Path(tempfile.mkdtemp()) / "experiment.slaw")
>>> print_directory(artifact)
├── diagnostics.json
├── draws.npz
├── fits.json
└── manifest.json

```

- `manifest.json` -- law, column roles, observed predictor ranges, design counts, provenance.
- `fits.json` -- point estimates (normalized and raw-scale), variance components, optimizer and
    goodness-of-fit diagnostics.
- `draws.npz` -- the uncertainty draws themselves, keyed `"<target>/<parameter>"`, plus `run_sd` and
    the observed run-level deviations.
- `diagnostics.json` -- correlations, cross-target comparisons, fit quality and every warning.

A reloaded model predicts identically, to the bit:

```pycon
>>> from simple_scaling_laws import load
>>> load(artifact).predict(points).equals(model.predict(points))
True

```

## Warnings

The package returns a fit whenever one is mathematically possible and records its objections instead
of raising them, so an automated caller can decide how much to trust a prediction:

```pycon
>>> thin = simulate_runs(
...     {"test_loss__ce": {"E": 1.0, "A": 2.0, "alpha": 0.3, "B": 1.5, "beta": 0.25}},
...     model_sizes=(1e6, 1e8),
...     dataset_sizes=(1e3, 1e5),
...     runs_per_config=1,
...     evaluations_per_run=4,
...     run_sd=0.02,
...     eval_sd=0.03,
...     seed=0,
... )
>>> for note in fit(thin, n_draws=50, seed=0).warnings:
...     print(note.code, "-", note.severity)
underdetermined - error
too_few_configurations - warning
too_few_predictor_levels - error
too_few_predictor_levels - error
single_run_per_configuration - warning
parametric_uncertainty - warning

```

Detected conditions include: too few distinct configurations, fewer configurations than parameters,
**a predictor measured at fewer than three distinct scales**, a predictor held fixed, collinear
predictors, a single run per configuration, no repeated evaluations, optimizer non-convergence,
parameters pinned to a bound, weakly identified exponents, parameters that trade off against each
other without changing the curve, lack of fit, a constant target, no
uncertainty information at all, and extrapolation at predict time.

The three-scales rule deserves its own note, because it is the one failure widening cannot rescue.
Restricted to a single predictor a power-law term is `E + A·x^-α`: three free parameters, so two
scales can be fit exactly by infinitely many different curves. (A law that puts more on a predictor
needs more scales of it, and the note says how many: four for `shifted-separable-power`, six for
`broken-power`.) A design with only two dataset
sizes yields parameters that look plausible, intervals that look reasonable, and simulated coverage
of **0-38%**. The package raises an `error`-severity `too_few_predictor_levels` note in **100%** of
such fits -- it cannot give you the right answer, but it will not pretend to.

## Programmatic inspection

```pycon
>>> {k: round(v, 3) for k, v in model.exponents("test_loss__cross_entropy").items()}
{'alpha': 0.305, 'beta': 0.25}
>>> low, high = model.conf_int("test_loss__cross_entropy")["alpha"]
>>> low < 0.3 < high
True
>>> model.observed_domain["model_size__n_params"]["max"]
100000000.0
>>> model.draws["test_loss__cross_entropy"].params.shape
(200, 5)

```

`model.fits`, `model.draws`, `model.diagnostics`, `model.observed_domain`, `model.predictors` and
`model.warnings` are all exposed. There is no plotting here by design: anything you would want to
draw can be built from `model.predict()`.

## Development

See [CONTRIBUTORS.md](https://github.com/mmcdermott/simple-scaling-laws/blob/main/CONTRIBUTORS.md). In short:

```bash
uv sync
uv run pre-commit install
uv run pytest -v
```

## License

MIT. See [LICENSE](https://github.com/mmcdermott/simple-scaling-laws/blob/main/LICENSE).
