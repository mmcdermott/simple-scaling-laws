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

    def raw_scale_params(self, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Restate fitted parameters in raw predictor units.

        Fitting happens on normalized predictors ``x / reference``. Exponents do not depend on that
        choice but amplitudes do, and *how* they depend on it is a property of the law's algebra --
        a separable amplitude absorbs one predictor's reference, a joint amplitude absorbs all of
        them -- so each law family owns its own conversion.

        Args:
            params: Parameter vector of length :attr:`n_params`, on normalized predictors.
            reference: Normalization constant for each predictor, shape ``(n_predictors,)``.

        Returns:
            A mapping from parameter name to its raw-scale value.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> raw = law.raw_scale_params(np.array([1.0, 2.0, 3.0, 0.5, 0.25]), np.array([100.0, 16.0]))
            >>> raw["A"], raw["B"], raw["alpha"]
            (20.0, 6.0, 0.5)

            Evaluating the raw-scale parameters at raw predictor values gives the same curve:

            >>> x = np.array([[400.0, 81.0]])
            >>> normalized = law.evaluate(np.array([1.0, 2.0, 3.0, 0.5, 0.25]), np.log(x / [100.0, 16.0]))
            >>> raw_vector = np.array([raw[name] for name in law.param_names])
            >>> bool(np.isclose(law.evaluate(raw_vector, np.log(x)), normalized)[0])
            True
        """
        return _RAW_SCALE[self.law](self, np.asarray(params, dtype=float), np.asarray(reference, dtype=float))

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


def _values(law: LawInstance, params: np.ndarray) -> dict[str, float]:
    """Map a parameter vector to its names, leaving every value on its current scale."""
    return dict(zip(law.param_names, (float(v) for v in params), strict=True))


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

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """``A_i`` on normalized predictors is ``A_i * reference_i**alpha_i`` on raw ones."""
        values = _values(law, params)
        exponents = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : law.n_linear]):
            values[name] *= float(reference[i] ** exponents[i])
        return values


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

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """The single amplitude absorbs every predictor's reference: ``A * prod_i ref_i**alpha_i``."""
        values = _values(law, params)
        values["A"] *= float(np.prod(reference ** params[law.n_linear :]))
        return values


class _SeparableInteractionPower:
    """``f(x) = E + sum_i A_i * x_i ** -alpha_i - C * prod_i x_i ** -gamma_i``.

    With one model-size predictor ``N`` and one dataset-size predictor ``D`` this is
    ``E + A N**-alpha + B D**-beta - C N**-gamma D**-delta``: the separable law plus one joint term
    that fades as either scale grows.

    **Why the joint term is subtracted.** The separable law forces the effect of model size to be
    the same at every dataset size. The observed failure of that assumption is one-sided: with
    scarce data a larger model can be *worse*, and the penalty disappears as data grow. Differentiating,
    ``d f / d log N = -alpha A N**-alpha + gamma C N**-gamma D**-delta``, so a subtracted joint term is
    what makes the second, positive, piece available. Writing the sign into the basis rather than
    freeing the sign of ``C`` keeps a loss's amplitudes non-negative, so the bounded linear solve --
    and the "irreducible floor plus decaying terms" reading of ``E``, ``A`` and ``B`` -- is unchanged.

    **What it costs.** Two exponents per predictor instead of one, so eight parameters where the
    separable law has five. The joint term is also exactly aliased with a main effect whenever its
    other exponents are zero (``C N**-gamma`` cancels against ``A N**-alpha`` at ``delta = 0``), so on
    data with no real interaction ``C`` and its exponents are not identified and land on a bound.
    That is reported as ``parameter_at_bound``; it is the signal to fall back to ``separable-power``,
    which this law contains exactly at ``C = 0``.

    With several predictors per role the main effects stay separable and the interaction is a single
    joint product over all of them.

    This form is a package-level extension motivated by the model/data coupling reported by Kaplan
    et al. (2020), *Scaling Laws for Neural Language Models*. It is not a law named in that paper.
    """

    name: ClassVar[str] = "separable-interaction-power"
    description: ClassVar[str] = (
        "E + sum_i A_i * x_i**-alpha_i - C * prod_i x_i**-gamma_i (separable terms minus a joint term)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        predictors = model_size + dataset_size
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        joint_exps = _names("gamma", model_size) + _names("delta", dataset_size)
        return LawInstance(
            law=_SeparableInteractionPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", *amps, "C", *exps, *joint_exps),
            param_kinds=(OFFSET, *([AMPLITUDE] * (len(amps) + 1)), *([EXPONENT] * (2 * len(predictors)))),
            # Every predictor carries two exponents: its main-effect one, then its joint-term one.
            exponent_predictor=tuple(range(len(predictors))) * 2,
        )

    @staticmethod
    def design(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1, x_1**-alpha_1, ..., x_k**-alpha_k, -prod_i x_i**-gamma_i]``."""
        k = len(law.predictors)
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1 : 1 + k] = np.exp(-log_x * exponents[None, :k])
        out[:, -1] = -np.exp(-(log_x * exponents[None, k:]).sum(axis=1))
        return out

    @staticmethod
    def many_designs(law: LawInstance, exponents: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many exponent vectors at once, shape ``(n_draws, n_points, n_linear)``."""
        k = len(law.predictors)
        out = np.empty((exponents.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1 : 1 + k] = np.exp(-exponents[:, None, :k] * log_x[None, :, :])
        out[:, :, -1] = -np.exp(-(exponents[:, None, k:] * log_x[None, :, :]).sum(axis=2))
        return out

    @staticmethod
    def exponent_grad(
        law: LawInstance, linear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """Main-effect exponents as in the separable law; joint ones scale the (negative) joint column."""
        k = len(law.predictors)
        grad = np.empty((log_x.shape[0], 2 * k), dtype=float)
        grad[:, :k] = -linear[None, 1 : 1 + k] * log_x * design[:, 1 : 1 + k]
        grad[:, k:] = -linear[-1] * log_x * design[:, -1:]
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Main amplitudes absorb their own predictor's reference; ``C`` absorbs all of them."""
        values = _values(law, params)
        k = len(law.predictors)
        exponents = params[law.n_linear :]
        for i, name in enumerate(law.param_names[1 : 1 + k]):
            values[name] *= float(reference[i] ** exponents[i])
        values["C"] *= float(np.prod(reference ** exponents[k:]))
        return values


_LAWS: dict[str, Any] = {
    law.name: law for law in (_SeparablePower, _MultiplicativePower, _SeparableInteractionPower)
}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_EXPONENT_GRADS = {name: law.exponent_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}
_RAW_SCALE = {name: law.raw_scale for name, law in _LAWS.items()}

#: Name of the law used when the user does not choose one.
DEFAULT_LAW = _SeparablePower.name


def available_laws() -> dict[str, str]:
    """Return a mapping from law name to a one-line description.

    Examples:
        >>> for name, description in available_laws().items():
        ...     print(f"{name}: {description}")
        separable-power: E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)
        multiplicative-power: E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)
        separable-interaction-power: E + sum_i A_i * x_i**-alpha_i - C * prod_i x_i**-gamma_i (separable
        terms minus a joint term)
    """
    return {name: law.description for name, law in _LAWS.items()}


def build_law(law: str, model_size: Sequence[str], dataset_size: Sequence[str]) -> LawInstance:
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

        The interaction law keeps the separable parameters and adds a joint amplitude with one more
        exponent per predictor:

        >>> build_law("separable-interaction-power", ["model_size__n"], ["dataset_size__d"]).param_names
        ('E', 'A', 'B', 'C', 'alpha', 'beta', 'gamma', 'delta')
        >>> build_law("no-such-law", ["model_size__n"], [])
        Traceback (most recent call last):
        ...
        simple_scaling_laws.laws.LawError: Unknown law 'no-such-law'; available: separable-power,
        multiplicative-power, separable-interaction-power
    """
    if law not in _LAWS:
        raise LawError(f"Unknown law {law!r}; available: {', '.join(_LAWS)}")
    model_size, dataset_size = tuple(model_size), tuple(dataset_size)
    if not model_size and not dataset_size:
        raise LawError(f"Law {law!r} needs at least one predictor column")
    return _LAWS[law].build(model_size, dataset_size)
