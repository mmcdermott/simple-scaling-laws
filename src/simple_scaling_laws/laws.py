"""The built-in scaling-law family registry.

Every law in this package has the same structure: it is **linear** in an offset and one or more amplitudes,
and **nonlinear** only in its exponents. That structure is what makes the fitter small and robust -- given the
exponents, the remaining parameters are solved exactly by (bounded) linear least squares, so the optimizer
only ever has to search a low-dimensional, well-behaved space.

Predictors are always supplied in *normalized* form (each divided by a reference scale, see
:mod:`simple_scaling_laws.data`), so amplitudes are on the target's own scale and the exponents are unchanged
by the normalization.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

MAX_EXPONENT: float = 3.0
OFFSET, AMPLITUDE, EXPONENT = "offset", "amplitude", "exponent"


class LawError(ValueError):
    """Raised for unknown laws or malformed law configurations."""


@dataclasses.dataclass(frozen=True, slots=True)
class LawInstance:
    """A scaling law bound to a concrete set of predictor columns."""

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
        return sum(k != EXPONENT for k in self.param_kinds)

    @property
    def n_exponents(self) -> int:
        return len(self.exponent_predictor)

    @property
    def display_names(self) -> tuple[str, ...]:
        names = list(self.param_names)
        linear = names[: self.n_linear]
        exponents = names[self.n_linear :]
        if len(linear) - 1 == len(exponents):
            ordered = [linear[0]]
            for amp, exp in zip(linear[1:], exponents, strict=True):
                ordered.extend([amp, exp])
            return tuple(ordered)
        return tuple(linear + exponents)

    def design(self, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        return _DESIGNS[self.law](self, np.asarray(exponents, dtype=float), log_x)

    def evaluate(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        exponents = params[self.n_linear :]
        return self.design(exponents, log_x) @ linear

    def evaluate_many(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.atleast_2d(np.asarray(params, dtype=float))
        linear = params[:, : self.n_linear]
        exponents = params[:, self.n_linear :]
        designs = _MANY_DESIGNS[self.law](self, exponents, log_x)
        return np.einsum("bl,bnl->bn", linear, designs)

    def jacobian(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        exponents = params[self.n_linear :]
        design = self.design(exponents, log_x)
        jac = np.empty((log_x.shape[0], self.n_params), dtype=float)
        jac[:, : self.n_linear] = design
        jac[:, self.n_linear :] = _EXPONENT_GRADS[self.law](self, linear, design, log_x)
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
    return [f"{base}_{_short(c)}" for c in columns]


def _raw_values(law: LawInstance, params: np.ndarray) -> dict[str, float]:
    return dict(zip(law.param_names, (float(v) for v in params), strict=True))


class _SeparablePower:
    name: ClassVar[str] = "separable-power"
    description: ClassVar[str] = "E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            law=_SeparablePower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", *amps, *exps),
            param_kinds=(OFFSET, *([AMPLITUDE] * len(amps)), *([EXPONENT] * len(exps))),
            exponent_predictor=tuple(range(len(model_size) + len(dataset_size))),
        )

    @staticmethod
    def design(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-log_x * exponents[None, :])
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((exponents.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-exponents[:, None, :] * log_x[None, :, :])
        return out

    @staticmethod
    def exponent_grad(law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        return -linear[None, 1:] * log_x * design[:, 1:]

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _raw_values(law, params)
        exponents = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : law.n_linear]):
            values[name] *= float(reference[i] ** exponents[i])
        return values


class _MultiplicativePower:
    name: ClassVar[str] = "multiplicative-power"
    description: ClassVar[str] = "E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            law=_MultiplicativePower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", *exps),
            param_kinds=(OFFSET, AMPLITUDE, *([EXPONENT] * len(exps))),
            exponent_predictor=tuple(range(len(model_size) + len(dataset_size))),
        )

    @staticmethod
    def design(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = np.exp(-(log_x * exponents[None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((exponents.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(-(exponents[:, None, :] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def exponent_grad(law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        return -linear[1] * log_x * design[:, 1:2]

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _raw_values(law, params)
        values["A"] *= float(np.prod(reference ** params[law.n_linear :]))
        return values


class _SeparableInteractionPower:
    """Separable powers with a model/data interaction that can penalize the low-data regime.

    For one model-size predictor ``N`` and one dataset-size predictor ``D`` this is
    ``E + A*N**-alpha + B*D**-beta - C*N**-gamma*D**-delta``. For losses, all three amplitudes are
    nonnegative. The negative interaction can therefore make larger models harmful at small ``D`` while
    disappearing at large ``D``. Metrics retain the package's existing signed-amplitude behavior.

    This is a generic power-law interaction motivated by the joint model/data dependence reported by
    Kaplan et al. (2020), not a named law from that paper.
    """

    name: ClassVar[str] = "separable-interaction-power"
    description: ClassVar[str] = (
        "E + A*N**-alpha + B*D**-beta - C*N**-gamma*D**-delta "
        "(model/data interaction; exactly one model and one dataset predictor)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        if len(model_size) != 1 or len(dataset_size) != 1:
            raise LawError("'separable-interaction-power' requires exactly one model-size and one dataset-size predictor")
        return LawInstance(
            law=_SeparableInteractionPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", "B", "C", "alpha", "beta", "gamma", "delta"),
            param_kinds=(OFFSET, AMPLITUDE, AMPLITUDE, AMPLITUDE, EXPONENT, EXPONENT, EXPONENT, EXPONENT),
            exponent_predictor=(0, 1, 0, 1),
        )

    @staticmethod
    def design(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        alpha, beta, gamma, delta = exponents
        log_n, log_d = log_x[:, 0], log_x[:, 1]
        out = np.empty((log_x.shape[0], 4), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = np.exp(-alpha * log_n)
        out[:, 2] = np.exp(-beta * log_d)
        out[:, 3] = -np.exp(-gamma * log_n - delta * log_d)
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        log_n, log_d = log_x[:, 0], log_x[:, 1]
        out = np.empty((exponents.shape[0], log_x.shape[0], 4), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(-exponents[:, 0, None] * log_n[None, :])
        out[:, :, 2] = np.exp(-exponents[:, 1, None] * log_d[None, :])
        out[:, :, 3] = -np.exp(
            -exponents[:, 2, None] * log_n[None, :] - exponents[:, 3, None] * log_d[None, :]
        )
        return out

    @staticmethod
    def exponent_grad(law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        grad = np.empty((log_x.shape[0], 4), dtype=float)
        grad[:, 0] = -linear[1] * log_x[:, 0] * design[:, 1]
        grad[:, 1] = -linear[2] * log_x[:, 1] * design[:, 2]
        # design[:, 3] is negative, so differentiating -C exp(-gamma log N - delta log D)
        # gives -C * design[:, 3] * log(scale).
        grad[:, 2] = -linear[3] * log_x[:, 0] * design[:, 3]
        grad[:, 3] = -linear[3] * log_x[:, 1] * design[:, 3]
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _raw_values(law, params)
        _, a, b, c, alpha, beta, gamma, delta = params
        values["A"] = float(a * reference[0] ** alpha)
        values["B"] = float(b * reference[1] ** beta)
        values["C"] = float(c * reference[0] ** gamma * reference[1] ** delta)
        return values


_LAWS: dict[str, Any] = {
    law.name: law for law in (_SeparablePower, _MultiplicativePower, _SeparableInteractionPower)
}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_EXPONENT_GRADS = {name: law.exponent_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}
_RAW_SCALE = {name: law.raw_scale for name, law in _LAWS.items()}

DEFAULT_LAW = _SeparablePower.name


def available_laws() -> dict[str, str]:
    """Return a mapping from law name to a one-line description."""
    return {name: law.description for name, law in _LAWS.items()}


def build_law(law: str, model_size: Sequence[str], dataset_size: Sequence[str]) -> LawInstance:
    """Instantiate a named law for a concrete set of predictor columns."""
    if law not in _LAWS:
        raise LawError(f"Unknown law {law!r}; available: {', '.join(_LAWS)}")
    model_size, dataset_size = tuple(model_size), tuple(dataset_size)
    if not model_size and not dataset_size:
        raise LawError(f"Law {law!r} needs at least one predictor column")
    return _LAWS[law].build(model_size, dataset_size)
