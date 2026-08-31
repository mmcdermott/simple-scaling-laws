# simple-scaling-laws

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/simple-scaling-laws)](https://pypi.org/project/simple-scaling-laws/)
[![Tests](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/simple-scaling-laws/actions/workflows/code-quality-main.yaml)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/simple-scaling-laws#license)

> [!WARNING]
> **Internal research tooling, largely AI-generated.** This package was written primarily by an AI
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
current    interpolation  2.854 [2.843, 2.867]
10x-model  extrapolation  2.365 [2.341, 2.389]
10x-data   extrapolation  2.486 [2.463, 2.508]

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

Two law families, both linear in their offset and amplitudes and nonlinear only in their exponents:

```pycon
>>> from simple_scaling_laws import available_laws
>>> for name, description in available_laws().items():
...     print(f"{name:<22} {description}")
separable-power        E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)
multiplicative-power   E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)

```

With one model-size and one dataset-size column, `separable-power` is the conventional
`E + A·N^-α + B·D^-β`.

Losses are fit with non-negative amplitudes, which gives the usual "irreducible floor plus decaying
terms" reading. Metrics get free-signed amplitudes so an increasing metric such as AUROC can approach
its asymptote from below:

```pycon
>>> auroc = model.params("test_metric__auroc")
>>> auroc["A"] < 0 and auroc["B"] < 0
True

```

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
their evaluation residuals is estimated and reported; only the part of evaluation noise that is *not*
shared is treated as averaging away when runs are compared.

**Uncertainty is a wild cluster bootstrap over configurations.** Residuals around the fitted curve
are leverage-corrected and re-signed with Webb's six-point weights, drawn once per *configuration* so
that a scale's shared lack of fit is preserved rather than averaged away. This was chosen because it
is assumption-free about the error distribution, stays valid with the four-to-a-dozen configurations
a real scaling experiment has, keeps the `(N, D)` design fixed instead of resampling it into
degeneracy, and is conservative: residuals carry the law's own misspecification into the intervals.
Draws -- not quantiles -- are stored, so any summary can be computed later.

**Correlations are computed across runs.** Loss/metric correlations use run-level means, with
intervals from resampling configurations. The naive evaluation-row correlation is reported alongside
purely so its inflation is visible:

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

## Prediction semantics

| `kind`    | answers                                                              |
| --------- | -------------------------------------------------------------------- |
| `mean`    | what does this training procedure achieve *on average* at this scale |
| `new-run` | what might *one* newly trained model achieve at this scale           |

`mean` is the default because it is the scaling law itself. `new-run` adds training-run
stochasticity, resampled from the observed run-to-run deviations rather than assumed Gaussian:

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
single_run_per_configuration - warning
parametric_uncertainty - warning

```

Detected conditions include: too few distinct configurations, fewer configurations than parameters,
a predictor held fixed, collinear predictors, a single run per configuration, no repeated
evaluations, optimizer non-convergence, parameters pinned to a bound, weakly identified exponents,
lack of fit, a constant target, and extrapolation at predict time.

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

See [CONTRIBUTORS.md](CONTRIBUTORS.md). In short:

```bash
uv sync
uv run pre-commit install
uv run pytest -v
```

## License

MIT. See [LICENSE](LICENSE).
