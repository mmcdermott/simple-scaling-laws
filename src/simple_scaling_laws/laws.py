"""The built-in scaling-law family registry.

Every law in this package has the same structure: it is **linear** in an offset and one or more
amplitudes, and **nonlinear** only in its exponents. That structure is what makes the fitter small
and robust -- given the exponents, the remaining parameters are solved exactly by (bounded) linear
least squares, so the optimizer only ever has to search a low-dimensional, well-behaved space.

Predictors are always supplied in *normalized* form (each divided by a reference scale, see
:mod:`simple_scaling_laws.data`), so amplitudes are on the target's own scale and the exponents are
unchanged by the normalization.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Upper bound on any fitted exponent. Empirical ML scaling exponents are well under 1; the bound
#: exists to keep the optimizer inside a numerically sane region, not to express a belief.
MAX_EXPONENT: float = 3.0

#: Parameter kinds, in the order they appear in a parameter vector.
OFFSET, AMPLITUDE, EXPONENT = "offset", "amplitude", "exponent"


class LawError(ValueError):
    """Raised for unknown laws or malformed law configurations."""


@dataclasses.dataclass(frozen=True, slots=True)
class LawInstance:
    """A scaling law bound to a concrete set of predictor columns.

    A parameter vector is ordered ``[offset, *amplitudes, *exponents]``: the linear block first,
    then the nonlinear block. :attr:`n_linear` splits the two.

    Attributes:
        law: Name of the law family.
        model_size: Model-size predictor columns, in order.
        dataset_size: Dataset-size predictor columns, in order.
        param_names: Parameter names, in parameter-vector order.
        param_kinds: Parameter kinds, in parameter-vector order.
        exponent_predictor: For each exponent, the index of the predictor it applies to.
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
        """Number of parameters that enter the law linearly (the offset plus the amplitudes)."""
        return sum(k != EXPONENT for k in self.param_kinds)

    @property
    def n_exponents(self) -> int:
        """Number of exponents."""
        return len(self.exponent_predictor)

    @property
    def display_names(self) -> tuple[str, ...]:
        """Parameter names in human reading order: offset, then (amplitude, exponent) pairs.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> law.param_names
            ('E', 'A', 'B', 'alpha', 'beta')
            >>> law.display_names
            ('E', 'A', 'alpha', 'B', 'beta')
        """
        names = list(self.param_names)
        linear = names[: self.n_linear]
        exponents = names[self.n_linear :]
        if len(linear) - 1 == len(exponents):  # one amplitude per exponent
            ordered = [linear[0]]
            for amp, exp in zip(linear[1:], exponents, strict=True):
                ordered.extend([amp, exp])
            return tuple(ordered)
        return tuple(linear + exponents)

    def design(self, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrix of the linear block, given the exponents.

        Args:
            exponents: Shape ``(n_exponents,)``.
            log_x: Natural log of the normalized predictors, shape ``(n_obs, n_predictors)``.

        Returns:
            Array of shape ``(n_obs, n_linear)`` whose product with the linear parameters is the
            law's value.
        """
        return _DESIGNS[self.law](self, np.asarray(exponents, dtype=float), log_x)

    def evaluate(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Evaluate the law.

        Args:
            params: Parameter vector of length :attr:`n_params`.
            log_x: Natural log of the normalized predictors, shape ``(n_obs, n_predictors)``.

        Returns:
            Predicted values, shape ``(n_obs,)``.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> log_x = np.log(np.array([[1.0, 1.0], [4.0, 1.0]]))
            >>> law.evaluate(np.array([1.0, 2.0, 3.0, 0.5, 0.25]), log_x).round(4)
            array([6., 5.])
        """
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        exponents = params[self.n_linear :]
        return self.design(exponents, log_x) @ linear

    def evaluate_many(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Evaluate the law for many parameter vectors at once.

        This is how prediction propagates uncertainty: every bootstrap draw is evaluated at every
        requested point in one vectorized pass.

        Args:
            params: Parameter vectors, shape ``(n_draws, n_params)``.
            log_x: Natural log of the normalized predictors, shape ``(n_points, n_predictors)``.

        Returns:
            Predicted values, shape ``(n_draws, n_points)``.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> log_x = np.log(np.array([[1.0, 1.0], [4.0, 1.0]]))
            >>> draws = np.array([[1.0, 2.0, 3.0, 0.5, 0.25], [1.0, 2.0, 3.0, 1.0, 0.25]])
            >>> law.evaluate_many(draws, log_x)
            array([[6. , 5. ],
                   [6. , 4.5]])
        """
        params = np.atleast_2d(np.asarray(params, dtype=float))
        linear = params[:, : self.n_linear]
        exponents = params[:, self.n_linear :]
        designs = _MANY_DESIGNS[self.law](self, exponents, log_x)
        return np.einsum("bl,bnl->bn", linear, designs)

    def jacobian(self, params: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of :meth:`evaluate` with respect to ``params``.

        Args:
            params: Parameter vector of length :attr:`n_params`.
            log_x: Natural log of the normalized predictors, shape ``(n_obs, n_predictors)``.

        Returns:
            Array of shape ``(n_obs, n_params)``.
        """
        params = np.asarray(params, dtype=float)
        linear = params[: self.n_linear]
        exponents = params[self.n_linear :]
        design = self.design(exponents, log_x)
        jac = np.empty((log_x.shape[0], self.n_params), dtype=float)
        jac[:, : self.n_linear] = design
        jac[:, self.n_linear :] = _EXPONENT_GRADS[self.law](self, linear, design, log_x)
        return jac

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
        """Rebuild from :meth:`to_dict` output.

        Examples:
            >>> law = build_law("multiplicative-power", ("model_size__n",), ("dataset_size__d",))
            >>> LawInstance.from_dict(law.to_dict()) == law
            True
        """
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
    """Name one parameter per column: the bare base name if unique, else base plus column suffix."""
    if len(columns) == 1:
        return [base]
    return [f"{base}_{_short(c)}" for c in columns]


class _SeparablePower:
    """``f(x) = E + sum_i A_i * x_i ** -alpha_i``.

    With one model-size and one dataset-size predictor this is the conventional
    ``E + A N**-alpha + B D**-beta`` form.
    """

    name: ClassVar[str] = "separable-power"
    description: ClassVar[str] = "E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
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
        """Columns ``[1, x_1**-alpha_1, ..., x_k**-alpha_k]``."""
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1:] = np.exp(-log_x * exponents[None, :])
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many exponent vectors at once, shape ``(n_draws, n_points, n_linear)``."""
        out = np.empty((exponents.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1:] = np.exp(-exponents[:, None, :] * log_x[None, :, :])
        return out

    @staticmethod
    def exponent_grad(
        law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """``d f / d alpha_j = -A_j * log(x_j) * x_j**-alpha_j``."""
        return -linear[None, 1:] * log_x * design[:, 1:]


class _MultiplicativePower:
    """``f(x) = E + A * prod_i x_i ** -alpha_i``.

    The joint-power alternative to :class:`_SeparablePower`: a single amplitude, one exponent per
    predictor, and no separate per-predictor asymptote.
    """

    name: ClassVar[str] = "multiplicative-power"
    description: ClassVar[str] = "E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
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
        """Columns ``[1, prod_i x_i**-alpha_i]``."""
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = np.exp(-(log_x * exponents[None, :]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many exponent vectors at once, shape ``(n_draws, n_points, 2)``."""
        out = np.empty((exponents.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(-(exponents[:, None, :] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def exponent_grad(
        law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """``d f / d alpha_j = -A * log(x_j) * prod_i x_i**-alpha_i``."""
        return -linear[1] * log_x * design[:, 1:2]


_LAWS: dict[str, Any] = {law.name: law for law in (_SeparablePower, _MultiplicativePower)}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_EXPONENT_GRADS = {name: law.exponent_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}

#: Name of the law used when the user does not choose one.
DEFAULT_LAW = _SeparablePower.name


def available_laws() -> dict[str, str]:
    """Return a mapping from law name to a one-line description.

    Examples:
        >>> for name, description in available_laws().items():
        ...     print(f"{name}: {description}")
        separable-power: E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)
        multiplicative-power: E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)
    """
    return {name: law.description for name, law in _LAWS.items()}


def build_law(
    law: str, model_size: Sequence[str], dataset_size: Sequence[str]
) -> LawInstance:
    """Instantiate a named law for a concrete set of predictor columns.

    Args:
        law: One of the names in :func:`available_laws`.
        model_size: Model-size predictor columns.
        dataset_size: Dataset-size predictor columns.

    Returns:
        The bound :class:`LawInstance`.

    Raises:
        LawError: If the law name is unknown or no predictors were supplied.

    Examples:
        >>> law = build_law("separable-power", ["model_size__n_params"], ["dataset_size__n_subj"])
        >>> law.param_names
        ('E', 'A', 'B', 'alpha', 'beta')
        >>> law.n_linear, law.n_exponents
        (3, 2)
        >>> build_law("multiplicative-power", ["model_size__n"], ["dataset_size__d"]).param_names
        ('E', 'A', 'alpha', 'beta')
        >>> build_law("no-such-law", ["model_size__n"], [])
        Traceback (most recent call last):
        ...
        simple_scaling_laws.laws.LawError: Unknown law 'no-such-law'; available: separable-power,
        multiplicative-power
    """
    if law not in _LAWS:
        raise LawError(f"Unknown law {law!r}; available: {', '.join(_LAWS)}")
    model_size, dataset_size = tuple(model_size), tuple(dataset_size)
    if not model_size and not dataset_size:
        raise LawError(f"Law {law!r} needs at least one predictor column")
    return _LAWS[law].build(model_size, dataset_size)
