# Scaling-law extension roadmap

This document describes proposed extensions to `simple-scaling-laws` for model/data interactions, very-small-data regimes, finite-data saturation, and non-monotonic scaling. It complements issue #7.

## Status

Sections 1-5 below are implemented, plus one joint law the roadmap did not propose (Skaling), and the
README documents each law. What follows is kept as the design record; where the implementation went a different way, this is how:

| Proposal                             | Outcome                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| 1. Interaction law                   | `separable-interaction-power`, any number of predictors                             |
| 2. Kaplan-style joint scaling        | `kaplan-joint-power`, with `rho = alpha_N / alpha_D`                                |
| 3. Shifted power (Zhai et al.)       | `shifted-separable-power`; needs four levels per predictor                          |
| 4. M4 / bounded small-data scaling   | `bounded-rational-power`, shape exponent fixed at 1 only                            |
| 5. Broken Neural Scaling Laws        | `broken-power`, one break, one predictor                                            |
| Internal refactor (`ParameterSpec`)  | `LawInstance.nonlinear_specs`: per-parameter bounds and seeds, law-owned            |
| Raw-scale conversion behind the law  | `LawInstance.raw_scale_params`                                                      |
| Diagnostics for weak identifiability | `ill_conditioned_fit`; law-specific `too_few_predictor_levels`                      |
| Skaling joint law (Videau et al.)    | `skaling-joint-power`; Kaplan with a free inner data exponent, separable at `k = 1` |

Still open:

- **Full M4** with a fitted shape exponent. It has no closed form, so it cannot use the exact linear solve
    that every current law relies on.
- **An explicit small-scale limit.** `bounded-rational-power` estimates its plateau `I` from the
    data. Section 4's suggestion of letting a target declare it (chance level for AUROC, say) would
    make the law usable on experiments that do not reach far enough down to see the bend.
- **A broken law in a model/data capacity ratio** (the end of section 5), and more than one break.

## Current laws

When this roadmap was written the package supported:

- `separable-power`: `E + sum_i A_i x_i^-alpha_i`.
- `multiplicative-power`: `E + A prod_i x_i^-alpha_i`.

The fitter is intentionally optimized around **variable projection**: conditional on nonlinear exponents, offsets and amplitudes are solved by bounded linear least squares. New law families should preserve this structure whenever practical.

## 1. Model/data interaction law

Recommended first addition:

```math
L(N,D) = E + A N^{-\alpha} + B D^{-\beta} - C N^{-\gamma}D^{-\delta},
```

with nonnegative `A`, `B`, and `C` for losses.

The negative interaction term allows the marginal effect of increasing model size to reverse when data are scarce:

```math
\frac{\partial L}{\partial \log N}
= -\alpha A N^{-\alpha} + \gamma C N^{-\gamma}D^{-\delta}.
```

At small `D`, the second term can dominate and make larger models worse; at large `D`, the interaction vanishes and the ordinary model-scaling term dominates.

This exact equation is a package-level extension, not a named law from the literature. Its motivation is the strong model/data coupling observed by Kaplan et al. (2020), especially the approximate collapse of overfitting as a function of `N^(alpha_N/alpha_D) / D`.

Implementation notes:

- Keep the interaction amplitude linear.
- Initially support one model-size and one dataset-size predictor, or generalize to a product interaction over all model/data predictors if property tests require arbitrary dimensionality.
- Add analytic Jacobian, batched evaluation, raw-scale conversion, simulation support, serialization, and recovery tests.
- Preserve deterministic bootstrap behavior.

## 2. Kaplan-style joint scaling

Kaplan et al. (2020) propose

```math
L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}.
```

A package-compatible version may add an irreducible floor:

```math
f(n,d)=E+A\left[n^{-\alpha_N/\alpha_D}+R d^{-1}\right]^{\alpha_D},
```

where `n` and `d` are normalized predictors.

This law captures model-limited and data-limited regimes and smooth saturation as model size grows at fixed finite data. It does **not** represent test performance getting worse as model size continues to increase.

Because `R` is a nonlinear scale parameter rather than an exponent, this law should be implemented only after the nonlinear-parameter metadata/seeding interface is generalized.

## 3. Shifted power (Zhai et al.)

A useful small-scale correction is

```math
f(n,d)=E+A(n+s_N)^{-\alpha}+B(d+s_D)^{-\beta}.
```

The shift prevents divergence as a predictor approaches zero and can fit observations before the clean asymptotic power-law regime. This is the single-predictor form `E = a(C+d)^{-b} + c` of Zhai et al. (2022). The form Alabdulmohsin et al. (2022) and Caballero et al. (2022) call M3 is a different one, Bansal et al.'s (2022) `L = alpha (D^{-1} + C)^p`, which shifts the reciprocal of the scale and so bounds the large-scale end rather than the small one.

The shifts `s_N` and `s_D` are nonlinear parameters but amplitudes remain linear conditional on them and the exponents.

## 4. M4 / bounded small-data scaling

Alabdulmohsin, Neyshabur, and Zhai (2022) propose

```math
\frac{\epsilon_x-\epsilon_\infty}{(\epsilon_0-\epsilon_x)^\alpha}=\beta x^c,
```

where `c < 0`, `epsilon_0` is the small-scale/random-guessing plateau and `epsilon_inf` the large-scale limit. The paper's `alpha` is a shape exponent, unrelated to the power-law exponents named `alpha` elsewhere in this package.

A multidimensional extension is

```math
\frac{f-E}{(I-f)^{\alpha_{M4}}}=g(n,d),
```

with a simple choice

```math
g(n,d)=A n^{-\alpha}+B d^{-\beta}.
```

For `alpha_M4 = 1`, the response is explicit:

```math
f=\frac{E+I g}{1+g}.
```

A good implementation sequence is therefore:

1. `bounded-rational-power` for the explicit `alpha_M4 = 1` special case;
2. full `m4-power` once nonlinear optimization and identifiability diagnostics are robust.

Do not silently infer the small-scale limit `I`. AUROC has a natural chance value of 0.5, but accuracy, AUPRC, and cross-entropy generally require task-specific information. Add target configuration for an explicit small-scale limit.

The existing automatic logit transform for `[0,1]` metrics must also be reconciled with M4's direct interpretation on the observed response scale.

## 5. Broken Neural Scaling Laws

Caballero et al. (2022) propose Broken Neural Scaling Laws (BNSL):

```math
y=a+b x^{-c_0}\prod_{i=1}^{K}\left(1+\left(\frac{x}{d_i}\right)^{1/f_i}\right)^{-c_i f_i}.
```

BNSLs can represent multiple slope regimes, delayed transitions, inflection points, and non-monotonic behavior including double descent.

Start conservatively with a single break (`K=1`) and strong identifiability diagnostics. A later model/data variant can define a capacity ratio

```math
z=N^\rho/D^\eta
```

and fit a broken law in `z`, motivated by Kaplan-style overfitting ratios and the movement of double-descent thresholds with dataset size.

## Required internal refactor

The current parameter representation assumes

```text
[offset, amplitudes..., exponents...]
```

with every nonlinear parameter treated as a nonnegative exponent. That assumption should be generalized before Kaplan/shifted/M4/BNSL support.

A law should eventually own:

1. parameter names and ordering;
2. which parameters are linear;
3. parameter-specific bounds;
4. nonlinear seed grids / initialization;
5. design-matrix construction when variable projection is available;
6. evaluation and batched evaluation;
7. Jacobian;
8. normalized-to-raw parameter conversion.

A minimal conceptual interface is:

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str
    linear: bool
    bounds: tuple[float, float]
    grid: tuple[float, ...] | None = None
```

The exact API is not prescribed; preserving variable projection and bootstrap efficiency is more important than the specific abstraction.

## Raw-scale conversion

`model.raw_scale_params()` currently special-cases the two original law names. Raw-scale conversion should move behind the law registry rather than accumulating conditionals in `model.py`.

Every law should have a test that normalized-parameter evaluation and raw-parameter evaluation agree at identical physical predictor values.

## Diagnostics

Richer laws should add diagnostics for:

- parameters at bounds;
- insufficient independent scale configurations;
- nearly singular local Jacobians / weak identifiability;
- interactions compatible with zero;
- crossovers or break locations outside the observed domain;
- bootstrap-refit failures;
- large disagreement between point estimates and bootstrap distributions.

Do not automatically prefer a richer law because its in-sample residual is lower. Extrapolation validation or explicit complexity penalties are more appropriate.

## Recommended PR sequence

1. **Interaction surface**: add the model/data interaction law with focused tests and law-owned raw-scale conversion.
2. **General nonlinear parameters**: refactor parameter metadata/seeding; add Kaplan and shifted laws.
3. **Bounded small-data laws**: add the rational `alpha_M4 = 1` law, then full M4 if stable.
4. **Broken laws**: add a conservatively scoped BNSL implementation after the simpler laws have been exercised on real data.

## References

- Hestness et al. (2017), *Deep Learning Scaling is Predictable, Empirically*, arXiv:1712.00409.
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models*, arXiv:2001.08361.
- Nakkiran et al. (2020), *Deep Double Descent: Where Bigger Models and More Data Hurt*, ICLR 2020.
- Zhai et al. (2022), *Scaling Vision Transformers*, CVPR 2022, arXiv:2106.04560.
- Bansal et al. (2022), *Data Scaling Laws in NMT: The Effect of Noise and Architecture*, ICML 2022, arXiv:2202.01994.
- Alabdulmohsin, Neyshabur, and Zhai (2022), *Revisiting Neural Scaling Laws in Language and Vision*, NeurIPS 2022.
- Caballero et al. (2022), *Broken Neural Scaling Laws*, arXiv:2210.14891.
- Videau, Youbi-Idrissi, Lopez-Paz and Ahuja (2026), *Skaling: Chinchilla's Exponents Meet Kaplan's Coupling*, arXiv:2608.07222.
- Bahri et al. (2024), *Explaining Neural Scaling Laws*, PNAS 121(27):e2311878121, arXiv:2102.06701.
