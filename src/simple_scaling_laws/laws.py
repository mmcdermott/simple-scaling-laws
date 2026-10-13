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

#: Upper bound on fitted power-law exponents.
MAX_EXPONENT: float = 3.0

#: Parameter kinds, in parameter-vector order.
OFFSET, AMPLITUDE, EXPONENT, SHIFT, SCALE = "offset", "amplitude", "exponent", "shift", "scale"
LINEAR_KINDS = frozenset((OFFSET, AMPLITUDE))


class LawError(ValueError):
    """Raised for unknown laws or malformed law configurations."""


@dataclasses.dataclass(frozen=True, slots=True)
class LawInstance:
    """A scaling law bound to a concrete set of predictor columns.

    Parameters are ordered with every linear parameter first, followed by the nonlinear block. The
    nonlinear block can contain exponents as well as shifts or scale ratios. ``exponent_predictor``
    records the predictor associated with each parameter whose kind is :data:`EXPONENT`.

    Attributes:
        law: Name of the law family.
        model_size: Model-size predictor columns, in order.
        dataset_size: Dataset-size predictor columns, in order.
        param_names: Parameter names, in parameter-vector order.
        param_kinds: Parameter kinds, in parameter-vector order.
        exponent_predictor: Predictor index for every true power-law exponent.
    """

    law: str
    model_size: tuple[str, ...]
    dataset_size: tuple[str, ...]
    param_names: tuple[str, ...]
    param_kinds: tuple[str, ...]
    exponent_predictor: tuple[int, ...]

    @property
    def predictors(self) -> tuple[str, ...]:
        """All predictor columns, model-size first."""
        return self.model_size + self.dataset_size

    @property
    def n_params(self) -> int:
        """Total number of free parameters."""
        return len(self.param_names)

    @property
    def n_linear(self) -> int:
        """Number of parameters entering the law linearly."""
        return sum(kind in LINEAR_KINDS for kind in self.param_kinds)

    @property
    def n_nonlinear(self) -> int:
        """Number of parameters in the nonlinear block."""
        return self.n_params - self.n_linear

    @property
    def n_exponents(self) -> int:
        """Number of true power-law exponents."""
        return sum(kind == EXPONENT for kind in self.param_kinds)

    @property
    def nonlinear_kinds(self) -> tuple[str, ...]:
        """Kinds in the nonlinear block, in optimization order."""
        return self.param_kinds[self.n_linear :]

    @property
    def display_names(self) -> tuple[str, ...]:
        """Parameter names in a stable human-readable order.

        Simple one-amplitude-per-exponent laws retain the historical interleaved display order. More
        general laws use their declared parameter order.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> law.param_names
            ('E', 'A', 'B', 'alpha', 'beta')
            >>> law.display_names
            ('E', 'A', 'alpha', 'B', 'beta')
        """
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
        """Design matrix of the linear block for fixed nonlinear parameters."""
        return _DESIGNS[self.law](self, np.asarray(nonlinear, dtype=float), log_x)

    def evaluate(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Evaluate the law for one parameter vector."""
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        nonlinear = params[self.n_linear :]
        return self.design(nonlinear, log_x) @ linear

    def evaluate_many(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Evaluate the law for many parameter vectors at once."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        linear = params[:, : self.n_linear]
        nonlinear = params[:, self.n_linear :]
        designs = _MANY_DESIGNS[self.law](self, nonlinear, log_x)
        return np.einsum("bl,bnl->bn", linear, designs)

    def jacobian(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of :meth:`evaluate` with respect to ``params``."""
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        nonlinear = params[self.n_linear :]
        design = self.design(nonlinear, log_x)
        jac = np.empty((log_x.shape[0], self.n_params), dtype=float)
        jac[:, : self.n_linear] = design
        jac[:, self.n_linear :] = _NONLINEAR_GRADS[self.law](
            self, linear, nonlinear, design, log_x
        )
        return jac

    def raw_scale_params(self, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Restate fitted parameters in raw predictor units."""
        return _RAW_SCALE[self.law](
            self,
            np.asarray(params, dtype=float),
            np.asarray(reference, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
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
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            law=data["law"],
            model_size=tuple(data["model_size"]),
            dataset_size=tuple(data["dataset_size"]),
            param_names=tuple(data["param_names"]),
            param_kinds=tuple(data["param_kinds"]),
            exponent_predictor=tuple(data["exponent_predictor"]),
        )


def _short(name: str) -> str:
    """Strip a ``role__`` prefix from a predictor column name."""
    _, sep, rest = name.partition("__")
    return rest if sep else name


def _names(base: str, columns: Sequence[str]) -> list[str]:
    """Name one parameter per column: bare base when unique, otherwise suffix by column."""
    if len(columns) == 1:
        return [base]
    return [f"{base}_{_short(column)}" for column in columns]


def _values(law: LawInstance, params: np.ndarray) -> dict[str, float]:
    """Map a parameter vector to names without changing scale."""
    return dict(zip(law.param_names, (float(value) for value in params), strict=True))


class _SeparablePower:
    """``f(x) = E + sum_i A_i * x_i ** -alpha_i``."""

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
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-log_x * nonlinear[None, :])
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-nonlinear[:, None, :] * log_x[None, :, :])
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance,
        linear: np.ndarray,
        nonlinear: np.ndarray,
        design: np.ndarray,
        log_x: np.ndarray,
    ) -> np.ndarray:
        del law, nonlinear
        return -linear[None, 1:] * log_x * design[:, 1:]

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _values(law, params)
        exponents = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : law.n_linear]):
            values[name] *= float(reference[i] ** exponents[i])
        return values


class _MultiplicativePower:
    """``f(x) = E + A * prod_i x_i ** -alpha_i``."""

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
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = np.exp(-(log_x * nonlinear[None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(-(nonlinear[:, None, :] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance,
        linear: np.ndarray,
        nonlinear: np.ndarray,
        design: np.ndarray,
        log_x: np.ndarray,
    ) -> np.ndarray:
        del law, nonlinear
        return -linear[1] * log_x * design[:, 1:2]

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _values(law, params)
        values["A"] *= float(np.prod(reference ** params[law.n_linear :]))
        return values


class _SeparableInteractionPower:
    """Separable main effects plus a low-scale joint interaction.

    For one model-size predictor ``N`` and one dataset-size predictor ``D`` the law is
    ``E + A*N**-alpha + B*D**-beta - C*N**-gamma*D**-delta``. With non-negative loss amplitudes,
    the negative joint term can make increasing model size harmful in low-data regimes while the
    interaction vanishes at large data scale.

    The exact equation is a package-level extension motivated by the model/data coupling reported by
    Kaplan et al. (2020), not a named law from that paper.
    """

    name: ClassVar[str] = "separable-interaction-power"
    description: ClassVar[str] = (
        "E + separable powers - C * joint power (model/data interaction with a low-scale penalty)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        predictors = model_size + dataset_size
        main_amps = _names("A", model_size) + _names("B", dataset_size)
        main_exps = _names("alpha", model_size) + _names("beta", dataset_size)
        interaction_exps = _names("gamma", model_size) + _names("delta", dataset_size)
        return LawInstance(
            law=_SeparableInteractionPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", *main_amps, "C", *main_exps, *interaction_exps),
            param_kinds=(
                OFFSET,
                *([AMPLITUDE] * len(main_amps)),
                AMPLITUDE,
                *([EXPONENT] * (2 * len(predictors))),
            ),
            exponent_predictor=tuple(range(len(predictors))) * 2,
        )

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        n_predictors = len(law.predictors)
        main = nonlinear[:n_predictors]
        interaction = nonlinear[n_predictors:]
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1 : 1 + n_predictors] = np.exp(-log_x * main[None, :])
        out[:, -1] = -np.exp(-(log_x * interaction[None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        n_predictors = len(law.predictors)
        main = nonlinear[:, :n_predictors]
        interaction = nonlinear[:, n_predictors:]
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1 : 1 + n_predictors] = np.exp(-main[:, None, :] * log_x[None, :, :])
        out[:, :, -1] = -np.exp(-(interaction[:, None, :] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance,
        linear: np.ndarray,
        nonlinear: np.ndarray,
        design: np.ndarray,
        log_x: np.ndarray,
    ) -> np.ndarray:
        del nonlinear
        n_predictors = len(law.predictors)
        grad = np.empty((log_x.shape[0], 2 * n_predictors), dtype=float)
        grad[:, :n_predictors] = -linear[None, 1 : 1 + n_predictors] * log_x * design[:, 1 : 1 + n_predictors]
        grad[:, n_predictors:] = -linear[-1] * log_x * design[:, -1, None]
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _values(law, params)
        n_predictors = len(law.predictors)
        nonlinear = params[law.n_linear :]
        main = nonlinear[:n_predictors]
        interaction = nonlinear[n_predictors:]
        for i, name in enumerate(law.param_names[1 : 1 + n_predictors]):
            values[name] *= float(reference[i] ** main[i])
        values["C"] *= float(np.prod(reference**interaction))
        return values


class _ShiftedSeparablePower:
    """Separable power law with a finite small-scale shift for every predictor.

    ``f(x) = E + sum_i A_i * (x_i + s_i)**-alpha_i`` on normalized predictors. Shifted power-law
    forms are commonly grouped as M3 in the neural-scaling literature; related forms appear in Zhai
    et al. (2022) and Bansal et al. (2022).
    """

    name: ClassVar[str] = "shifted-separable-power"
    description: ClassVar[str] = "E + sum_i A_i * (x_i + s_i)**-alpha_i (shifted/M3-style powers)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        predictors = model_size + dataset_size
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        shifts = _names("s", predictors)
        return LawInstance(
            law=_ShiftedSeparablePower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", *amps, *exps, *shifts),
            param_kinds=(
                OFFSET,
                *([AMPLITUDE] * len(amps)),
                *([EXPONENT] * len(predictors)),
                *([SHIFT] * len(predictors)),
            ),
            exponent_predictor=tuple(range(len(predictors))),
        )

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        n = len(law.predictors)
        exponents, shifts = nonlinear[:n], nonlinear[n:]
        base = np.exp(log_x) + shifts[None, :]
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-np.log(base) * exponents[None, :])
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        n = len(law.predictors)
        exponents, shifts = nonlinear[:, :n], nonlinear[:, n:]
        base = np.exp(log_x)[None, :, :] + shifts[:, None, :]
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-np.log(base) * exponents[:, None, :])
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance,
        linear: np.ndarray,
        nonlinear: np.ndarray,
        design: np.ndarray,
        log_x: np.ndarray,
    ) -> np.ndarray:
        n = len(law.predictors)
        exponents, shifts = nonlinear[:n], nonlinear[n:]
        base = np.exp(log_x) + shifts[None, :]
        powers = design[:, 1:]
        grad = np.empty((log_x.shape[0], 2 * n), dtype=float)
        grad[:, :n] = -linear[None, 1:] * np.log(base) * powers
        grad[:, n:] = -linear[None, 1:] * exponents[None, :] * powers / base
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
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
    """Kaplan-style joint model/data scaling with a package-level irreducible floor.

    On normalized model/data scales ``n`` and ``d`` this is
    ``E + A * (n**-rho + R*d**-1)**beta``. It is algebraically the Kaplan et al. (2020) joint form
    up to scale reparameterization, with the additional package-level ``E`` and ``A``. The original
    exponents are ``alpha_D = beta`` and ``alpha_N = rho * beta``.
    """

    name: ClassVar[str] = "kaplan-joint-power"
    description: ClassVar[str] = "E + A * (N**-rho + R/D)**beta (Kaplan-style joint model/data law)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        if len(model_size) != 1 or len(dataset_size) != 1:
            raise LawError("'kaplan-joint-power' requires exactly one model-size and one dataset-size predictor")
        return LawInstance(
            law=_KaplanJointPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", "rho", "beta", "R"),
            param_kinds=(OFFSET, AMPLITUDE, EXPONENT, EXPONENT, SCALE),
            exponent_predictor=(0, 1),
        )

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        del law
        rho, beta, ratio = nonlinear
        n_term = np.exp(-rho * log_x[:, 0])
        d_term = np.exp(-log_x[:, 1])
        joint = n_term + ratio * d_term
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = joint**beta
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        del law
        rho, beta, ratio = nonlinear[:, 0], nonlinear[:, 1], nonlinear[:, 2]
        n_term = np.exp(-rho[:, None] * log_x[None, :, 0])
        d_term = np.exp(-log_x[None, :, 1])
        joint = n_term + ratio[:, None] * d_term
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = joint**beta[:, None]
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance,
        linear: np.ndarray,
        nonlinear: np.ndarray,
        design: np.ndarray,
        log_x: np.ndarray,
    ) -> np.ndarray:
        del law
        rho, beta, ratio = nonlinear
        n_term = np.exp(-rho * log_x[:, 0])
        d_term = np.exp(-log_x[:, 1])
        joint = n_term + ratio * d_term
        fitted_basis = design[:, 1]
        amp = linear[1]
        grad = np.empty((log_x.shape[0], 3), dtype=float)
        grad[:, 0] = -amp * fitted_basis * beta * n_term * log_x[:, 0] / joint
        grad[:, 1] = amp * fitted_basis * np.log(joint)
        grad[:, 2] = amp * fitted_basis * beta * d_term / joint
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        values = _values(law, params)
        _, amp, rho, beta, ratio = params
        model_reference, data_reference = reference
        values["A"] = float(amp * model_reference ** (rho * beta))
        values["R"] = float(ratio * data_reference / model_reference**rho)
        return values


_LAW_TYPES = (
    _SeparablePower,
    _MultiplicativePower,
    _SeparableInteractionPower,
    _ShiftedSeparablePower,
    _KaplanJointPower,
)
_LAWS: dict[str, Any] = {law.name: law for law in _LAW_TYPES}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_NONLINEAR_GRADS = {name: law.nonlinear_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}
_RAW_SCALE = {name: law.raw_scale for name, law in _LAWS.items()}

#: Name of the law used when the user does not choose one.
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
