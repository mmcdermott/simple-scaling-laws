"""The built-in scaling-law family registry.

Every law in this package keeps its offset and amplitudes linear whenever possible. Conditional on
its nonlinear parameters, the remaining coefficients can therefore be solved exactly by bounded
linear least squares. This variable-projection structure is the main reason fitting remains stable
with the small scaling grids the package targets.

Predictors are always supplied in normalized form (each divided by a reference scale; see
:mod:`simple_scaling_laws.data`). Law families own the algebra for converting fitted parameters back
to raw predictor units.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

MAX_EXPONENT: float = 3.0

OFFSET = "offset"
AMPLITUDE = "amplitude"
EXPONENT = "exponent"
SIGNED_EXPONENT = "signed_exponent"
SHIFT = "shift"
SCALE = "scale"
POSITIVE_SCALE = "positive_scale"
LINEAR_KINDS = frozenset((OFFSET, AMPLITUDE))
EXPONENT_KINDS = frozenset((EXPONENT, SIGNED_EXPONENT))


class LawError(ValueError):
    """Raised for unknown laws or malformed law configurations."""


@dataclasses.dataclass(frozen=True, slots=True)
class LawInstance:
    """A scaling law bound to a concrete set of predictor columns.

    Parameters are ordered with every linear parameter first, followed by the nonlinear block.
    ``exponent_predictor`` records the predictor associated with every ordinary or signed scaling
    exponent.
    """

    law: str
    model_size: tuple[str, ...]
    dataset_size: tuple[str, ...]
    param_names: tuple[str, ...]
    param_kinds: tuple[str, ...]
    exponent_predictor: tuple[int, ...]

    @property
    def predictors(self) -> tuple[str, ...]:
        return self.model_size + self.dataset_size

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def n_linear(self) -> int:
        return sum(kind in LINEAR_KINDS for kind in self.param_kinds)

    @property
    def n_nonlinear(self) -> int:
        return self.n_params - self.n_linear

    @property
    def n_exponents(self) -> int:
        return sum(kind in EXPONENT_KINDS for kind in self.param_kinds)

    @property
    def nonlinear_kinds(self) -> tuple[str, ...]:
        return self.param_kinds[self.n_linear :]

    @property
    def display_names(self) -> tuple[str, ...]:
        names = list(self.param_names)
        linear = names[: self.n_linear]
        nonlinear = names[self.n_linear :]
        if len(linear) - 1 == len(nonlinear) and all(kind == EXPONENT for kind in self.nonlinear_kinds):
            ordered = [linear[0]]
            for amp, exponent in zip(linear[1:], nonlinear, strict=True):
                ordered.extend([amp, exponent])
            return tuple(ordered)
        return tuple(names)

    def design(self, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        return _DESIGNS[self.law](self, np.asarray(nonlinear, dtype=float), log_x)

    def evaluate(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        return self.design(params[self.n_linear :], log_x) @ params[: self.n_linear]

    def evaluate_many(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.atleast_2d(np.asarray(params, dtype=float))
        designs = _MANY_DESIGNS[self.law](self, params[:, self.n_linear :], log_x)
        return np.einsum("bl,bnl->bn", params[:, : self.n_linear], designs)

    def jacobian(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        nonlinear = params[self.n_linear :]
        design = self.design(nonlinear, log_x)
        jac = np.empty((log_x.shape[0], self.n_params), dtype=float)
        jac[:, : self.n_linear] = design
        jac[:, self.n_linear :] = _NONLINEAR_GRADS[self.law](self, linear, nonlinear, design, log_x)
        return jac

    def raw_scale_params(self, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        return _RAW_SCALE[self.law](self, np.asarray(params, dtype=float), np.asarray(reference, dtype=float))

    def to_dict(self) -> dict[str, Any]:
        return {
            "law": self.law,
            "model_size": list(self.model_size),
            "dataset_size": list(self.dataset_size),
            "param_names": list(self.param_names),
            "param_kinds": list(self.param_kinds),
            "exponent_predictor": list(self.exponent_predictor),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LawInstance:
        return cls(
            law=data["law"],
            model_size=tuple(data["model_size"]),
            dataset_size=tuple(data["dataset_size"]),
            param_names=tuple(data["param_names"]),
            param_kinds=tuple(data["param_kinds"]),
            exponent_predictor=tuple(data["exponent_predictor"]),
        )


def _short(name: str) -> str:
    _, sep, rest = name.partition("__")
    return rest if sep else name


def _names(base: str, columns: Sequence[str]) -> list[str]:
    if len(columns) == 1:
        return [base]
    return [f"{base}_{_short(column)}" for column in columns]


def _values(law: LawInstance, params: np.ndarray) -> dict[str, float]:
    return dict(zip(law.param_names, (float(value) for value in params), strict=True))


class _SeparablePower:
    name: ClassVar[str] = "separable-power"
    description: ClassVar[str] = "E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            _SeparablePower.name,
            model_size,
            dataset_size,
            ("E", *amps, *exps),
            (OFFSET, *([AMPLITUDE] * len(amps)), *([EXPONENT] * len(exps))),
            tuple(range(len(model_size) + len(dataset_size))),
        )

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((log_x.shape[0], law.n_linear))
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-log_x * nonlinear[None, :])
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear))
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-nonlinear[:, None, :] * log_x[None, :, :])
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del law, nonlinear
        return -linear[None, 1:] * log_x * design[:, 1:]

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        exponents = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : law.n_linear]):
            values[name] *= float(reference[i] ** exponents[i])
        return values


class _MultiplicativePower:
    name: ClassVar[str] = "multiplicative-power"
    description: ClassVar[str] = "E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)"

    @staticmethod
    def build(model_size, dataset_size):
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            _MultiplicativePower.name,
            model_size,
            dataset_size,
            ("E", "A", *exps),
            (OFFSET, AMPLITUDE, *([EXPONENT] * len(exps))),
            tuple(range(len(model_size) + len(dataset_size))),
        )

    @staticmethod
    def design(law, nonlinear, log_x):
        out = np.empty((log_x.shape[0], 2))
        out[:, 0] = 1.0
        out[:, 1] = np.exp(-(log_x * nonlinear[None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2))
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(-(nonlinear[:, None, :] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del law, nonlinear
        return -linear[1] * log_x * design[:, 1:2]

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        values["A"] *= float(np.prod(reference ** params[law.n_linear :]))
        return values


class _SeparableInteractionPower:
    name: ClassVar[str] = "separable-interaction-power"
    description: ClassVar[str] = "E + separable powers - C * joint power (model/data interaction with a low-scale penalty)"

    @staticmethod
    def build(model_size, dataset_size):
        predictors = model_size + dataset_size
        main_amps = _names("A", model_size) + _names("B", dataset_size)
        main_exps = _names("alpha", model_size) + _names("beta", dataset_size)
        interaction_exps = _names("gamma", model_size) + _names("delta", dataset_size)
        return LawInstance(
            _SeparableInteractionPower.name,
            model_size,
            dataset_size,
            ("E", *main_amps, "C", *main_exps, *interaction_exps),
            (OFFSET, *([AMPLITUDE] * len(main_amps)), AMPLITUDE, *([EXPONENT] * (2 * len(predictors)))),
            tuple(range(len(predictors))) * 2,
        )

    @staticmethod
    def design(law, nonlinear, log_x):
        n = len(law.predictors)
        out = np.empty((log_x.shape[0], law.n_linear))
        out[:, 0] = 1.0
        out[:, 1 : 1 + n] = np.exp(-log_x * nonlinear[:n][None, :])
        out[:, -1] = -np.exp(-(log_x * nonlinear[n:][None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        n = len(law.predictors)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear))
        out[:, :, 0] = 1.0
        out[:, :, 1 : 1 + n] = np.exp(-nonlinear[:, None, :n] * log_x[None, :, :])
        out[:, :, -1] = -np.exp(-(nonlinear[:, None, n:] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del nonlinear
        n = len(law.predictors)
        grad = np.empty((log_x.shape[0], 2 * n))
        grad[:, :n] = -linear[None, 1 : 1 + n] * log_x * design[:, 1 : 1 + n]
        grad[:, n:] = -linear[-1] * log_x * design[:, -1, None]
        return grad

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        n = len(law.predictors)
        nonlinear = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : 1 + n]):
            values[name] *= float(reference[i] ** nonlinear[i])
        values["C"] *= float(np.prod(reference ** nonlinear[n:]))
        return values


class _ShiftedSeparablePower:
    name: ClassVar[str] = "shifted-separable-power"
    description: ClassVar[str] = "E + sum_i A_i * (x_i + s_i)**-alpha_i (shifted/M3-style powers)"

    @staticmethod
    def build(model_size, dataset_size):
        predictors = model_size + dataset_size
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        shifts = _names("s", predictors)
        return LawInstance(
            _ShiftedSeparablePower.name,
            model_size,
            dataset_size,
            ("E", *amps, *exps, *shifts),
            (OFFSET, *([AMPLITUDE] * len(amps)), *([EXPONENT] * len(predictors)), *([SHIFT] * len(predictors))),
            tuple(range(len(predictors))),
        )

    @staticmethod
    def design(law, nonlinear, log_x):
        n = len(law.predictors)
        exponents, shifts = nonlinear[:n], nonlinear[n:]
        base = np.exp(log_x) + shifts[None, :]
        out = np.empty((log_x.shape[0], law.n_linear))
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-np.log(base) * exponents[None, :])
        return out

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        n = len(law.predictors)
        exponents, shifts = nonlinear[:, :n], nonlinear[:, n:]
        base = np.exp(log_x)[None, :, :] + shifts[:, None, :]
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear))
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-np.log(base) * exponents[:, None, :])
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        n = len(law.predictors)
        exponents, shifts = nonlinear[:n], nonlinear[n:]
        base = np.exp(log_x) + shifts[None, :]
        powers = design[:, 1:]
        grad = np.empty((log_x.shape[0], 2 * n))
        grad[:, :n] = -linear[None, 1:] * np.log(base) * powers
        grad[:, n:] = -linear[None, 1:] * exponents[None, :] * powers / base
        return grad

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        n = len(law.predictors)
        nonlinear = params[law.n_linear :]
        exponents, shifts = nonlinear[:n], nonlinear[n:]
        for i, name in enumerate(law.param_names[1 : 1 + n]):
            values[name] *= float(reference[i] ** exponents[i])
        for i, name in enumerate(law.param_names[law.n_linear + n :]):
            values[name] = float(shifts[i] * reference[i])
        return values


class _KaplanJointPower:
    name: ClassVar[str] = "kaplan-joint-power"
    description: ClassVar[str] = "E + A * (N**-rho + R/D)**beta (Kaplan-style joint model/data law)"

    @staticmethod
    def build(model_size, dataset_size):
        if len(model_size) != 1 or len(dataset_size) != 1:
            raise LawError("'kaplan-joint-power' requires exactly one model-size and one dataset-size predictor")
        return LawInstance(
            _KaplanJointPower.name,
            model_size,
            dataset_size,
            ("E", "A", "rho", "beta", "R"),
            (OFFSET, AMPLITUDE, EXPONENT, EXPONENT, SCALE),
            (0, 1),
        )

    @staticmethod
    def design(law, nonlinear, log_x):
        del law
        rho, beta, ratio = nonlinear
        joint = np.exp(-rho * log_x[:, 0]) + ratio * np.exp(-log_x[:, 1])
        return np.column_stack((np.ones(log_x.shape[0]), joint**beta))

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        del law
        rho, beta, ratio = nonlinear[:, 0], nonlinear[:, 1], nonlinear[:, 2]
        joint = np.exp(-rho[:, None] * log_x[None, :, 0]) + ratio[:, None] * np.exp(-log_x[None, :, 1])
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2))
        out[:, :, 0] = 1.0
        out[:, :, 1] = joint ** beta[:, None]
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del law
        rho, beta, ratio = nonlinear
        n_term = np.exp(-rho * log_x[:, 0])
        d_term = np.exp(-log_x[:, 1])
        joint = n_term + ratio * d_term
        basis, amp = design[:, 1], linear[1]
        grad = np.empty((log_x.shape[0], 3))
        grad[:, 0] = -amp * basis * beta * n_term * log_x[:, 0] / joint
        grad[:, 1] = amp * basis * np.log(joint)
        grad[:, 2] = amp * basis * beta * d_term / joint
        return grad

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        _, amp, rho, beta, ratio = params
        values["A"] = float(amp * reference[0] ** (rho * beta))
        values["R"] = float(ratio * reference[1] / reference[0] ** rho)
        return values


class _BoundedRationalPower:
    name: ClassVar[str] = "bounded-rational-power"
    description: ClassVar[str] = "(E + I*g)/(1 + g), g=sum_i A_i*x_i**-alpha_i (bounded M4 lambda=1)"

    @staticmethod
    def build(model_size, dataset_size):
        predictors = model_size + dataset_size
        scales = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            _BoundedRationalPower.name,
            model_size,
            dataset_size,
            ("E", "I", *scales, *exps),
            (OFFSET, OFFSET, *([SCALE] * len(predictors)), *([EXPONENT] * len(predictors))),
            tuple(range(len(predictors))),
        )

    @staticmethod
    def _components(law, nonlinear, log_x):
        n = len(law.predictors)
        scales, exponents = nonlinear[:n], nonlinear[n:]
        return scales, np.exp(-log_x * exponents[None, :])

    @staticmethod
    def design(law, nonlinear, log_x):
        scales, powers = _BoundedRationalPower._components(law, nonlinear, log_x)
        g = (powers * scales[None, :]).sum(axis=1)
        denominator = 1.0 + g
        return np.column_stack((1.0 / denominator, g / denominator))

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        n = len(law.predictors)
        scales, exponents = nonlinear[:, :n], nonlinear[:, n:]
        powers = np.exp(-exponents[:, None, :] * log_x[None, :, :])
        g = (powers * scales[:, None, :]).sum(axis=2)
        denominator = 1.0 + g
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2))
        out[:, :, 0] = 1.0 / denominator
        out[:, :, 1] = g / denominator
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del design
        n = len(law.predictors)
        scales, powers = _BoundedRationalPower._components(law, nonlinear, log_x)
        g = (powers * scales[None, :]).sum(axis=1)
        factor = (linear[1] - linear[0]) / (1.0 + g) ** 2
        grad = np.empty((log_x.shape[0], 2 * n))
        grad[:, :n] = factor[:, None] * powers
        grad[:, n:] = -factor[:, None] * scales[None, :] * log_x * powers
        return grad

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        n = len(law.predictors)
        nonlinear = params[law.n_linear :]
        scales, exponents = nonlinear[:n], nonlinear[n:]
        for i, name in enumerate(law.param_names[law.n_linear : law.n_linear + n]):
            values[name] = float(scales[i] * reference[i] ** exponents[i])
        return values


class _BrokenPower:
    """One-break Broken Neural Scaling Law for a single scale coordinate.

    ``f(x) = E + A*x**-c0*(1 + (x/d)**(1/f))**(-c1*f)``. The initial slope ``c0`` is
    non-negative, while the slope change ``c1`` may have either sign. A sufficiently negative
    ``c1`` can therefore reverse the asymptotic log-log slope, representing a genuine regime
    turnover rather than only monotone saturation.

    This is the ``K=1`` specialization of Caballero et al. (2022), *Broken Neural Scaling Laws*.
    Multiple breaks are deliberately deferred because they require substantially more scale points
    to identify reliably.
    """

    name: ClassVar[str] = "broken-power"
    description: ClassVar[str] = "one-break BNSL: E + A*x^-c0*(1+(x/d)^(1/f))^(-c1*f)"

    @staticmethod
    def build(model_size, dataset_size):
        if len(model_size) + len(dataset_size) != 1:
            raise LawError("'broken-power' requires exactly one scale predictor")
        return LawInstance(
            _BrokenPower.name,
            model_size,
            dataset_size,
            ("E", "A", "c0", "c1", "d", "f"),
            (OFFSET, AMPLITUDE, EXPONENT, SIGNED_EXPONENT, POSITIVE_SCALE, POSITIVE_SCALE),
            (0, 0),
        )

    @staticmethod
    def _basis(nonlinear, log_x):
        c0, c1, d, width = nonlinear
        log_scale = log_x[:, 0]
        z = (log_scale - np.log(d)) / width
        softplus = np.logaddexp(0.0, z)
        return np.exp(-c0 * log_scale - c1 * width * softplus), z, softplus

    @staticmethod
    def design(law, nonlinear, log_x):
        del law
        basis, _, _ = _BrokenPower._basis(nonlinear, log_x)
        return np.column_stack((np.ones(log_x.shape[0]), basis))

    @staticmethod
    def many_designs(law, nonlinear, log_x):
        del law
        c0, c1, d, width = (nonlinear[:, i] for i in range(4))
        log_scale = log_x[None, :, 0]
        z = (log_scale - np.log(d)[:, None]) / width[:, None]
        log_basis = -c0[:, None] * log_scale - c1[:, None] * width[:, None] * np.logaddexp(0.0, z)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2))
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(log_basis)
        return out

    @staticmethod
    def nonlinear_grad(law, linear, nonlinear, design, log_x):
        del law
        c0, c1, d, width = nonlinear
        log_scale = log_x[:, 0]
        _, z, softplus = _BrokenPower._basis(nonlinear, log_x)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        weighted_basis = linear[1] * design[:, 1]
        grad = np.empty((log_x.shape[0], 4))
        grad[:, 0] = -weighted_basis * log_scale
        grad[:, 1] = -weighted_basis * width * softplus
        grad[:, 2] = weighted_basis * c1 * sigmoid / d
        grad[:, 3] = -weighted_basis * c1 * (softplus - sigmoid * z)
        return grad

    @staticmethod
    def raw_scale(law, params, reference):
        values = _values(law, params)
        _, amp, c0, _, d, _ = params
        values["A"] = float(amp * reference[0] ** c0)
        values["d"] = float(d * reference[0])
        return values


_LAW_TYPES = (
    _SeparablePower,
    _MultiplicativePower,
    _SeparableInteractionPower,
    _ShiftedSeparablePower,
    _KaplanJointPower,
    _BoundedRationalPower,
    _BrokenPower,
)
_LAWS: dict[str, Any] = {law.name: law for law in _LAW_TYPES}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_NONLINEAR_GRADS = {name: law.nonlinear_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}
_RAW_SCALE = {name: law.raw_scale for name, law in _LAWS.items()}

DEFAULT_LAW = _SeparablePower.name


def available_laws() -> dict[str, str]:
    return {name: law.description for name, law in _LAWS.items()}


def build_law(law: str, model_size: Sequence[str], dataset_size: Sequence[str]) -> LawInstance:
    if law not in _LAWS:
        raise LawError(f"Unknown law {law!r}; available: {', '.join(_LAWS)}")
    model_size, dataset_size = tuple(model_size), tuple(dataset_size)
    if not model_size and not dataset_size:
        raise LawError(f"Law {law!r} needs at least one predictor column")
    return _LAWS[law].build(model_size, dataset_size)
