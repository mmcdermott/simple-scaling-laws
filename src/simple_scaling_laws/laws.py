"""The built-in scaling-law family registry.

Every law in this package has the same structure: it is **linear** in a small block of parameters -- an
offset and one or more amplitudes, or for ``bounded-rational-power`` its two limits -- and **nonlinear** in
everything else. That structure is what makes the fitter small and robust -- given the
nonlinear parameters, the remaining ones are solved exactly by (bounded) linear least squares, so the
optimizer only ever has to search a low-dimensional, well-behaved space.

For the original laws the nonlinear block is nothing but exponents. Richer laws add other kinds of nonlinear
parameter -- an additive shift, a ratio between two terms -- and those do not share an exponent's natural
range. So each law states, per nonlinear parameter, the box it may move in and the values worth starting the
optimizer from (:class:`NonlinearSpec`), and may consult the observed predictors to do it. The fitter itself
knows nothing about any particular kind.

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

#: Candidate exponent values used to seed the optimizer, spanning the empirically plausible range.
EXPONENT_GRID: tuple[float, ...] = (0.01, 0.03, 0.08, 0.2, 0.5, 1.2)

#: Upper bound on a ratio between two terms of a law, in normalized units. Predictors are normalized
#: to a geometric mean of one, so a ratio of one means the two terms matter equally at the centre of
#: the design; at a hundred, one of them is invisible across any design this package targets.
MAX_SCALE: float = 100.0

#: Candidate ratios used to seed the optimizer, centred on "equally important at the centre".
SCALE_GRID: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

#: Bounds assumed for data-dependent parameters when no predictors are supplied: a design spanning
#: one decade either side of its centre.
DEFAULT_LOG_HALF_RANGE: float = float(np.log(10.0))

#: Parameter kinds. The linear kinds come first in a parameter vector, then the nonlinear ones.
OFFSET, AMPLITUDE, EXPONENT = "offset", "amplitude", "exponent"
#: An additive shift inside a power term, ``(x + s) ** -alpha``, in normalized predictor units.
SHIFT = "shift"
#: A non-negative ratio weighting one term against another inside a nonlinearity.
SCALE = "scale"
#: A change in log-log slope across a break. Unlike a scaling exponent it may be negative.
SIGNED_EXPONENT = "signed_exponent"
#: The scale at which a law changes regime, in normalized predictor units.
LOCATION = "location"
#: How many e-folds of scale a change of regime is spread over.
WIDTH = "width"
#: The kinds that are exponents, and so are reported by ``ScalingLawModel.exponents``. Most are slopes on
#: log-log axes; in the joint laws some are factors of one (Kaplan's ``rho``, Skaling's ``k``), and the
#: slope in a resource is then a product, which each law's docstring gives.
EXPONENT_KINDS: frozenset[str] = frozenset({EXPONENT, SIGNED_EXPONENT})
#: The kinds that enter a law linearly and are therefore solved exactly rather than searched.
LINEAR_KINDS: frozenset[str] = frozenset({OFFSET, AMPLITUDE})


class LawError(ValueError):
    """Raised for unknown laws or malformed law configurations."""


@dataclasses.dataclass(frozen=True, slots=True)
class NonlinearSpec:
    """Where one nonlinear parameter may go, and where to start looking for it.

    Attributes:
        lower: Lower bound.
        upper: Upper bound.
        grid: Starting values for the optimizer's grid of restarts, all within the bounds.
        resolution: For a parameter measured on the predictors' own scale -- a break location or a
            shift -- the smallest difference that matters, here the smallest observed scale. A value
            then counts as sitting on a bound only when it is within a tiny fraction of
            ``max(|bound|, resolution)`` of it. ``None`` keeps the default, a fraction of the box's
            width, which is right for exponents and ratios but flags a break 1.7 decades inside an
            eight-decade sweep as pinned.
    """

    lower: float
    upper: float
    grid: tuple[float, ...]
    resolution: float | None = None


#: The specification shared by every ordinary scaling exponent: non-negative, so a term decays as
#: scale grows, and capped at :data:`MAX_EXPONENT`.
EXPONENT_SPEC = NonlinearSpec(0.0, MAX_EXPONENT, EXPONENT_GRID)

#: The specification of a ratio between two terms.
SCALE_SPEC = NonlinearSpec(0.0, MAX_SCALE, SCALE_GRID)

#: A coupling exponent (``skaling-joint-power``'s ``k``): bounded like any exponent, and seeded across
#: the range Videau et al. (2026) measured and at one, where the law is separable.
COUPLING_SPEC = NonlinearSpec(0.0, MAX_EXPONENT, (0.3, 0.6, 1.0))

#: The specification of a change in slope: as large as an exponent may be, in either direction.
#: Seeded at no change, a moderate one and a pronounced one each way.
SLOPE_CHANGE_SPEC = NonlinearSpec(-MAX_EXPONENT, MAX_EXPONENT, (-1.0, -0.3, 0.0, 0.3, 1.0))


@dataclasses.dataclass(frozen=True, slots=True)
class LawInstance:
    """A scaling law bound to a concrete set of predictor columns.

    A parameter vector is ordered ``[*offsets, *amplitudes, *nonlinear]``: the linear block first,
    then the nonlinear block. There is usually one offset, the large-scale limit; a law with a limit
    at each end has two. :attr:`n_linear` splits the two. For the simplest laws the nonlinear
    block is only exponents; :attr:`nonlinear_kinds` says what it holds in general.

    Attributes:
        law: Name of the law family.
        model_size: Model-size predictor columns, in order.
        dataset_size: Dataset-size predictor columns, in order.
        param_names: Parameter names, in parameter-vector order.
        param_kinds: Parameter kinds, in parameter-vector order.
        exponent_predictor: For each exponent (each parameter of a kind in :data:`EXPONENT_KINDS`), the
            index of the predictor it is filed under.
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
        return sum(k in LINEAR_KINDS for k in self.param_kinds)

    @property
    def n_nonlinear(self) -> int:
        """Number of parameters the optimizer has to search over."""
        return self.n_params - self.n_linear

    @property
    def nonlinear_kinds(self) -> tuple[str, ...]:
        """The kind of each nonlinear parameter, in parameter-vector order."""
        return self.param_kinds[self.n_linear :]

    @property
    def n_exponents(self) -> int:
        """Number of slopes on log-log axes -- not every nonlinear parameter is one.

        Examples:
            >>> law = build_law("shifted-separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> law.n_nonlinear, law.n_exponents
            (4, 2)
        """
        return len(self.exponent_predictor)

    @property
    def min_predictor_levels(self) -> tuple[int, ...]:
        """How many distinct values each predictor must take before its terms are identified.

        A predictor measured at ``L`` values offers ``L - 1`` contrasts, and an additive term in
        that predictor is identified only if it has no more free parameters than that. A plain power
        term ``A * x ** -alpha`` has two, hence three levels; a shifted one has three, hence four.

        Examples:
            >>> build_law("separable-power", ("model_size__n",), ("dataset_size__d",)).min_predictor_levels
            (3, 3)
            >>> build_law("shifted-separable-power", ("model_size__n",), ()).min_predictor_levels
            (4,)
        """
        return _MIN_LEVELS[self.law](self)

    def nonlinear_specs(self, log_x: np.ndarray | None = None) -> tuple[NonlinearSpec, ...]:
        """Bounds and starting values for each nonlinear parameter.

        Args:
            log_x: Log-normalized predictors actually observed, shape ``(n_obs, n_predictors)``. A
                law whose parameters only mean something relative to the design -- a shift is large
                or small compared with the scales that were run -- uses it to place them. Without
                it, a design spanning a decade either side of its centre is assumed.

        Returns:
            One :class:`NonlinearSpec` per nonlinear parameter, in parameter-vector order.

        Examples:
            >>> law = build_law("separable-power", ("model_size__n",), ("dataset_size__d",))
            >>> law.nonlinear_specs() == (EXPONENT_SPEC, EXPONENT_SPEC)
            True

            A shift is bounded by the largest scale observed, so it widens with the design:

            >>> shifted = build_law("shifted-separable-power", ("model_size__n",), ())
            >>> narrow = shifted.nonlinear_specs(np.log(np.array([[0.1], [1.0], [10.0]])))
            >>> wide = shifted.nonlinear_specs(np.log(np.array([[0.01], [1.0], [100.0]])))
            >>> round(narrow[1].upper, 6), round(wide[1].upper, 6)
            (10.0, 100.0)
        """
        return _SPECS[self.law](self, None if log_x is None else np.asarray(log_x, dtype=float))

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
        nonlinear = names[self.n_linear :]
        paired = all(kind == EXPONENT for kind in self.nonlinear_kinds)
        if paired and len(linear) - 1 == len(nonlinear):  # one amplitude per exponent
            ordered = [linear[0]]
            for amp, exp in zip(linear[1:], nonlinear, strict=True):
                ordered.extend([amp, exp])
            return tuple(ordered)
        return tuple(linear + nonlinear)

    def design(self, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrix of the linear block, given the nonlinear parameters.

        Args:
            nonlinear: Shape ``(n_nonlinear,)``.
            log_x: Natural log of the normalized predictors, shape ``(n_obs, n_predictors)``.

        Returns:
            Array of shape ``(n_obs, n_linear)`` whose product with the linear parameters is the
            law's value.
        """
        return _DESIGNS[self.law](self, np.asarray(nonlinear, dtype=float), log_x)

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
        nonlinear = params[self.n_linear :]
        return self.design(nonlinear, log_x) @ linear

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
        nonlinear = params[:, self.n_linear :]
        designs = _MANY_DESIGNS[self.law](self, nonlinear, log_x)
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
        nonlinear = params[self.n_linear :]
        design = self.design(nonlinear, log_x)
        jac = np.empty((log_x.shape[0], self.n_params), dtype=float)
        jac[:, : self.n_linear] = design
        jac[:, self.n_linear :] = _NONLINEAR_GRADS[self.law](self, linear, nonlinear, design, log_x)
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

        The law is rebuilt from its name and predictors and checked against what was stored, so an
        artifact from a law that no longer exists, or whose parameters have since changed, fails here
        with a :class:`LawError` rather than later with a wrong answer.

        Raises:
            LawError: If the stored law is unknown or does not match the current definition.

        Examples:
            >>> law = build_law("multiplicative-power", ("model_size__n",), ("dataset_size__d",))
            >>> LawInstance.from_dict(law.to_dict()) == law
            True
            >>> stale = {**law.to_dict(), "param_names": ["E", "A", "a", "b"]}
            >>> LawInstance.from_dict(stale)
            Traceback (most recent call last):
            ...
            simple_scaling_laws.laws.LawError: Stored law 'multiplicative-power' does not match its current
            definition: parameters ('E', 'A', 'a', 'b') were saved, ('E', 'A', 'alpha', 'beta') are
            defined now
        """
        stored = cls(
            law=data["law"],
            model_size=tuple(data["model_size"]),
            dataset_size=tuple(data["dataset_size"]),
            param_names=tuple(data["param_names"]),
            param_kinds=tuple(data["param_kinds"]),
            exponent_predictor=tuple(data["exponent_predictor"]),
        )
        current = build_law(stored.law, stored.model_size, stored.dataset_size)
        if current != stored:
            raise LawError(
                f"Stored law {stored.law!r} does not match its current definition: parameters "
                f"{stored.param_names} were saved, {current.param_names} are defined now"
            )
        return stored


def _short(name: str) -> str:
    """Strip a ``role__`` prefix from a predictor column name."""
    _, sep, rest = name.partition("__")
    return rest if sep else name


def _names(base: str, columns: Sequence[str]) -> list[str]:
    """Name one parameter per column: the bare base name if unique, else base plus column suffix."""
    if len(columns) == 1:
        return [base]
    return [f"{base}_{_short(c)}" for c in columns]


def _role_names(base: str, model_size: Sequence[str], dataset_size: Sequence[str]) -> list[str]:
    """Like :func:`_names` over both roles at once, naming the role too where suffixes alone would clash.

    Examples:
        >>> _role_names("s", ["model_size__n_params"], ["dataset_size__n_subjects"])
        ['s_n_params', 's_n_subjects']
        >>> _role_names("s", ["model_size__n"], ["dataset_size__n"])
        ['s_model_size_n', 's_dataset_size_n']
    """
    names = _names(base, [*model_size, *dataset_size])
    if len(set(names)) == len(names):
        return names
    return [f"{base}_{column.replace('__', '_')}" for column in (*model_size, *dataset_size)]


def _exponent_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
    """Specifications for a law whose nonlinear block is nothing but exponents."""
    return (EXPONENT_SPEC,) * law.n_nonlinear


def _three_levels(law: LawInstance) -> tuple[int, ...]:
    """Three levels of every predictor, the floor for a term that needs both a slope and a bend."""
    return (3,) * len(law.predictors)


def _log_range(log_x: np.ndarray | None, column: int) -> tuple[float, float]:
    """Smallest and largest observed log-normalized value of one predictor."""
    if log_x is None or log_x.size == 0:
        return -DEFAULT_LOG_HALF_RANGE, DEFAULT_LOG_HALF_RANGE
    return float(log_x[:, column].min()), float(log_x[:, column].max())


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

    nonlinear_specs = staticmethod(_exponent_specs)
    min_levels = staticmethod(_three_levels)

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
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
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

    nonlinear_specs = staticmethod(_exponent_specs)
    min_levels = staticmethod(_three_levels)

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
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """``d f / d alpha_j = -A * log(x_j) * prod_i x_i**-alpha_i``."""
        return -linear[1] * log_x * design[:, 1:2]

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """The single amplitude absorbs every predictor's reference: ``A * prod_i ref_i**alpha_i``."""
        values = _values(law, params)
        values["A"] *= float(np.prod(reference ** params[law.n_linear :]))
        return values


class _ShiftedSeparablePower:
    """``f(x) = E + sum_i A_i * (x_i + s_i) ** -alpha_i``.

    A plain power term diverges as its predictor goes to zero, and is a straight line on log-log
    axes all the way down. Real curves are not: at the small end of an experiment the loss flattens
    toward a finite value before the clean power-law regime begins. A non-negative shift ``s`` inside
    the power gives the term a finite value ``A * s ** -alpha`` at zero scale and leaves its
    large-scale behavior untouched, so the pre-asymptotic points stop dragging the exponent down.

    The single-predictor form is the saturating law ``E = a * (C + d) ** -b + c`` that Zhai et al.
    (2022), *Scaling Vision Transformers*, fit to error against compute, where the shift ``d`` gives a
    zero-compute model a finite error. A related but different shifted form, ``alpha * (D ** -1 + C)
    ** p`` from Bansal et al. (2022), *Data Scaling Laws in NMT*, shifts the *reciprocal* of the scale
    and so bounds the large-scale end instead; that one is what Alabdulmohsin et al. (2022) and
    Caballero et al. (2022) label **M3**, and it is not this law. The separable sum over several
    predictors is this package's arrangement.

    **What it costs.** One more parameter per predictor, and a fourth level of every predictor: with
    only three there are two contrasts from which to learn an amplitude, an exponent *and* a shift.
    At ``s = 0`` the law is exactly ``separable-power``, so on data that are already a clean power law
    the shifts sit on their lower bound and are reported as ``parameter_at_bound``.

    The shift is in normalized predictor units and is bounded above by the largest scale observed:
    beyond that the term is nearly flat across the whole experiment and the shift can no longer be
    told apart from the amplitude and the exponent.
    """

    name: ClassVar[str] = "shifted-separable-power"
    description: ClassVar[str] = (
        "E + sum_i A_i * (x_i + s_i)**-alpha_i (power terms with a finite small-scale limit)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        predictors = model_size + dataset_size
        amps = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        shifts = _role_names("s", model_size, dataset_size)
        return LawInstance(
            law=_ShiftedSeparablePower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", *amps, *exps, *shifts),
            param_kinds=(
                OFFSET,
                *([AMPLITUDE] * len(amps)),
                *([EXPONENT] * len(exps)),
                *([SHIFT] * len(shifts)),
            ),
            exponent_predictor=tuple(range(len(predictors))),
        )

    @staticmethod
    def nonlinear_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
        """Exponents as usual; each shift between zero and the largest scale its predictor reached.

        The starting values are the three qualitatively different cases: no shift at all, a shift
        comparable to the smallest scale observed (it bends only the low end), and one comparable to
        the centre of the design (it bends everything).
        """
        k = len(law.predictors)
        shifts = []
        for column in range(k):
            low, high = _log_range(log_x, column)
            upper = float(np.exp(high))
            seeds = (0.0, min(float(np.exp(low)), upper), min(1.0, upper))
            shifts.append(NonlinearSpec(0.0, upper, seeds, resolution=float(np.exp(low))))
        return (*([EXPONENT_SPEC] * k), *shifts)

    @staticmethod
    def min_levels(law: LawInstance) -> tuple[int, ...]:
        """An amplitude, an exponent and a shift per predictor need three contrasts, so four levels."""
        return (4,) * len(law.predictors)

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1, (x_1 + s_1)**-alpha_1, ..., (x_k + s_k)**-alpha_k]``."""
        k = len(law.predictors)
        out = np.empty((log_x.shape[0], law.n_linear), dtype=float)
        out[:, 0] = 1.0
        out[:, 1:] = (np.exp(log_x) + nonlinear[None, k:]) ** -nonlinear[None, :k]
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many nonlinear vectors at once, shape ``(n_draws, n_points, n_linear)``."""
        k = len(law.predictors)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], law.n_linear), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1:] = (np.exp(log_x)[None, :, :] + nonlinear[:, None, k:]) ** -nonlinear[:, None, :k]
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """With ``u = x + s``: ``d/d alpha = -A log(u) u**-alpha`` and ``d/d s = -A alpha u**-alpha / u``."""
        k = len(law.predictors)
        base = np.exp(log_x) + nonlinear[None, k:]
        terms = linear[None, 1:] * design[:, 1:]
        grad = np.empty((log_x.shape[0], 2 * k), dtype=float)
        grad[:, :k] = -terms * np.log(base)
        grad[:, k:] = -terms * nonlinear[None, :k] / base
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Amplitudes absorb ``reference**alpha``; a shift is a length, so it simply scales."""
        values = _values(law, params)
        k = len(law.predictors)
        exponents = params[law.n_linear : law.n_linear + k]
        for i, name in enumerate(law.param_names[1 : 1 + k]):
            values[name] *= float(reference[i] ** exponents[i])
        for i, name in enumerate(law.param_names[law.n_linear + k :]):
            values[name] *= float(reference[i])
        return values


class _KaplanJointPower:
    """``f(N, D) = E + A * (N ** -rho + R / D) ** beta``.

    The joint model/data law of Kaplan et al. (2020), *Scaling Laws for Neural Language Models*
    (their equation 1.5), ``L = [(N_c / N) ** (alpha_N / alpha_D) + D_c / D] ** alpha_D``, rewritten so
    that the fitter's linear block is visible. Multiplying their bracket through by a constant turns
    ``N_c`` into the amplitude ``A`` and leaves one ratio ``R`` between the two terms; ``rho`` is
    their ``alpha_N / alpha_D`` and ``beta`` their ``alpha_D``. The offset ``E`` is this package's
    addition -- Kaplan et al. fit a loss with no irreducible floor -- and a fit is free to put it at
    zero.

    What it describes that the separable law cannot is a *bottleneck*: whichever resource is scarcer
    sets the loss. With ample data the bracket is ``N ** -rho`` and the law is a power law in model
    size with exponent ``rho * beta``; with an ample model it is ``R / D`` and the law is a power law
    in data with exponent ``beta``. In between, adding the abundant resource does almost nothing --
    the curve in model size saturates at a level set by the data. It is monotone in both: it does not
    describe a larger model being *worse*; ``broken-power`` can, along one axis.

    It takes exactly one model-size and one dataset-size predictor, because the form is specifically
    a contest between those two. At ``R = 0`` data drop out altogether and only the product
    ``rho * beta`` is identified, which is reported as ``parameter_at_bound`` and
    ``ill_conditioned_fit``.
    """

    name: ClassVar[str] = "kaplan-joint-power"
    description: ClassVar[str] = "E + A * (N**-rho + R / D)**beta (Kaplan et al. joint model/data law)"

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        if len(model_size) != 1 or len(dataset_size) != 1:
            raise LawError(
                f"Law {_KaplanJointPower.name!r} needs exactly one model-size and one dataset-size "
                f"predictor, got {len(model_size)} and {len(dataset_size)}"
            )
        return LawInstance(
            law=_KaplanJointPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", "rho", "beta", "R"),
            param_kinds=(OFFSET, AMPLITUDE, EXPONENT, EXPONENT, SCALE),
            # ``rho`` shapes the model-size term. ``beta`` is the exponent the law has in data once
            # data are the bottleneck, so it is filed under the dataset-size predictor.
            exponent_predictor=(0, 1),
        )

    @staticmethod
    def nonlinear_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
        """Two exponents and the ratio between the model term and the data term."""
        return (EXPONENT_SPEC, EXPONENT_SPEC, SCALE_SPEC)

    min_levels = staticmethod(_three_levels)

    @staticmethod
    def _bracket(nonlinear: np.ndarray, log_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The model term ``N**-rho`` and the whole bracket ``N**-rho + R / D``."""
        model_term = np.exp(-nonlinear[..., 0, None] * log_x[:, 0])
        return model_term, model_term + nonlinear[..., 2, None] * np.exp(-log_x[:, 1])

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1, (N**-rho + R / D)**beta]``."""
        _, bracket = _KaplanJointPower._bracket(nonlinear, log_x)
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = bracket ** nonlinear[1]
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many nonlinear vectors at once, shape ``(n_draws, n_points, 2)``."""
        _, bracket = _KaplanJointPower._bracket(nonlinear, log_x)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = bracket ** nonlinear[:, 1, None]
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """With bracket ``g``: ``d/d rho = -A beta g**(beta-1) N**-rho log N``, and likewise for ``R``."""
        model_term, bracket = _KaplanJointPower._bracket(nonlinear, log_x)
        value = linear[1] * design[:, 1]
        slope = value * nonlinear[1] / bracket
        grad = np.empty((log_x.shape[0], 3), dtype=float)
        grad[:, 0] = -slope * model_term * log_x[:, 0]
        grad[:, 1] = value * np.log(bracket)
        grad[:, 2] = slope * np.exp(-log_x[:, 1])
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Pull ``N_ref**rho`` out of the bracket: it rescales ``A`` by its ``beta`` power, and ``R``."""
        values = _values(law, params)
        _, _, rho, beta, _ = params
        values["A"] *= float(reference[0] ** (rho * beta))
        values["R"] *= float(reference[1] / reference[0] ** rho)
        return values


class _SkalingJointPower:
    """``f(N, D) = E + A * (N ** -alpha + R * D ** -beta) ** k``.

    The **Skaling** law of Videau, Youbi-Idrissi, Lopez-Paz & Ahuja (2026), *Skaling: Chinchilla's
    Exponents Meet Kaplan's Coupling* (arXiv:2608.07222), ``L = (A' / N ** alpha + B' / D ** beta) ** k
    + E``, rewritten so that the fitter's linear block is visible: pulling ``A'`` out of the bracket
    leaves the amplitude ``A = A' ** k`` and the ratio ``R = B' / A'``. Their ``A'`` is therefore
    ``A ** (1 / k)`` and their ``B'`` is ``R * A ** (1 / k)``.

    It is Chinchilla's separable law with one *coupling exponent* ``k`` applied to the sum of its two
    power terms. At ``k = 1`` it is exactly ``separable-power``. With the data exponent inside the
    bracket fixed at one it is ``kaplan-joint-power``, which is the special case that ties the inner
    exponents together. Its mixed derivative in ``log N`` and ``log D`` has the sign of ``k * (k - 1)``:
    for ``0 < k < 1`` each resource makes the other more valuable -- the negative interaction the
    authors measured on large language-model grids, with ``k`` between about 0.3 and 0.9 -- while the
    loss still falls monotonically in both. With ample data the law is a power law in model size with
    exponent ``alpha * k``; with an ample model, a power law in data with exponent ``beta * k``.

    **What it costs.** One parameter more than the separable law. ``k`` is identified by how the two
    terms bend into each other where neither dominates -- at the edges of a grid only the products
    ``alpha * k`` and ``beta * k`` are seen -- so the paper fit 117 to 404 runs per grid. Simulated
    3x3 grids with two runs per configuration were enough here: ``k`` came back within about 0.01 of
    the truth at 0.5 and 0.8, and nominal 95% prediction intervals covered 88-100%. On data with no
    coupling the fit puts ``k`` at one -- ``separable-power``, reached as an interior point rather than
    a bound -- and raises no warning, which is the honest answer and not a failure to identify.

    It takes exactly one model-size and one dataset-size predictor, as in the paper. With one
    predictor the bracket collapses to ``A * N ** -(alpha * k)`` and ``k`` would not be identified at
    all.
    """

    name: ClassVar[str] = "skaling-joint-power"
    description: ClassVar[str] = (
        "E + A * (N**-alpha + R * D**-beta)**k (Videau et al. Skaling law: coupled model/data terms)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        if len(model_size) != 1 or len(dataset_size) != 1:
            raise LawError(
                f"Law {_SkalingJointPower.name!r} needs exactly one model-size and one dataset-size "
                f"predictor, got {len(model_size)} and {len(dataset_size)}"
            )
        return LawInstance(
            law=_SkalingJointPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", "alpha", "beta", "k", "R"),
            param_kinds=(OFFSET, AMPLITUDE, EXPONENT, EXPONENT, EXPONENT, SCALE),
            # ``k`` multiplies both slopes; as in ``kaplan-joint-power`` the outer exponent is filed
            # under the dataset-size predictor.
            exponent_predictor=(0, 1, 1),
        )

    @staticmethod
    def nonlinear_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
        """Two inner exponents, the coupling exponent, and the ratio between the two terms.

        The coupling exponent gets three seeds rather than the usual six, spread over the range the paper
        measured and the separable value of one; the inner exponents and the ratio keep the usual grids.
        """
        return (EXPONENT_SPEC, EXPONENT_SPEC, COUPLING_SPEC, SCALE_SPEC)

    min_levels = staticmethod(_three_levels)

    @staticmethod
    def _bracket(nonlinear: np.ndarray, log_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The model term, the data term, and the bracket ``N**-alpha + R * D**-beta``."""
        model_term = np.exp(-nonlinear[..., 0, None] * log_x[:, 0])
        data_term = np.exp(-nonlinear[..., 1, None] * log_x[:, 1])
        return model_term, data_term, model_term + nonlinear[..., 3, None] * data_term

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1, (N**-alpha + R * D**-beta)**k]``."""
        _, _, bracket = _SkalingJointPower._bracket(nonlinear, log_x)
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = bracket ** nonlinear[2]
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many nonlinear vectors at once, shape ``(n_draws, n_points, 2)``."""
        _, _, bracket = _SkalingJointPower._bracket(nonlinear, log_x)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = bracket ** nonlinear[:, 2, None]
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """With bracket ``g``: ``d/d alpha = -A k g**(k-1) N**-alpha log N``, ``d/d k = A g**k log g``."""
        model_term, data_term, bracket = _SkalingJointPower._bracket(nonlinear, log_x)
        value = linear[1] * design[:, 1]
        slope = value * nonlinear[2] / bracket
        grad = np.empty((log_x.shape[0], 4), dtype=float)
        grad[:, 0] = -slope * model_term * log_x[:, 0]
        grad[:, 1] = -slope * nonlinear[3] * data_term * log_x[:, 1]
        grad[:, 2] = value * np.log(bracket)
        grad[:, 3] = slope * data_term
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """Pull ``N_ref**alpha`` out of the bracket: it rescales ``A`` by its ``k`` power, and ``R``."""
        values = _values(law, params)
        _, _, alpha, beta, k, _ = params
        values["A"] *= float(reference[0] ** (alpha * k))
        values["R"] *= float(reference[1] ** beta / reference[0] ** alpha)
        return values


class _BoundedRationalPower:
    """``f(x) = (E + I * g(x)) / (1 + g(x))`` with ``g(x) = sum_i A_i * x_i ** -alpha_i``.

    Every other law here is unbounded at small scale: a power term grows without limit as its
    predictor shrinks. A real loss does not. A classifier trained on almost nothing sits at chance,
    and stays there -- a plateau ``I`` at the small end, as real as the floor ``E`` at the large one.
    This law is a weighted average of the two, ``f = E / (1 + g) + I * g / (1 + g)``, with the weight
    ``g`` an ordinary separable sum of power terms: large ``g`` (small scale) gives ``I``, small ``g``
    (large scale) gives ``E``, and in between ``f - E`` falls off as a power law.

    It is a special case of the **M4** estimator of Alabdulmohsin, Neyshabur & Zhai (2022),
    *Revisiting Neural Scaling Laws in Language and Vision*, whose defining relation is
    ``(eps_x - eps_inf) / (eps_0 - eps_x) ** alpha = beta * x ** c`` with ``c < 0``. Here ``eps_inf``
    is ``E``, ``eps_0`` is ``I``, the right-hand side is ``g``, and the paper's shape exponent -- its
    ``alpha``, unrelated to this package's power-law exponents -- is fixed at one. Only then can the
    relation be solved for ``f`` in closed form, and only then is ``f`` linear in ``(E, I)`` given
    ``g`` -- which is what lets the fitter solve both limits exactly instead of searching for them.
    Summing ``g`` over several predictors is this package's arrangement; the paper's is a single
    power term. The name records the form actually fit rather than claiming the full family.

    **What it costs, and when not to use it.** One more parameter than the separable law, the
    plateau. It is identified only if the experiment reaches far enough down to see the curve bend
    toward it: on data that are still a clean power law, ``I`` runs off to its bound while the
    ``A_i`` shrink to compensate, since ``I * A * x ** -alpha`` is all such data determine. That is
    reported as ``parameter_at_bound`` or ``ill_conditioned_fit``, and is the cue to use
    ``separable-power``.

    Both limits are estimated from the data; neither is assumed. For a metric confined to ``[0, 1]``
    the law is fit on the logit like every other, so ``E`` and ``I`` are then limits in logit units --
    finite, and already inside the metric's range by construction.
    """

    name: ClassVar[str] = "bounded-rational-power"
    description: ClassVar[str] = (
        "(E + I * g) / (1 + g), g = sum_i A_i * x_i**-alpha_i (plateau I at small scale, floor E at large)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        predictors = model_size + dataset_size
        weights = _names("A", model_size) + _names("B", dataset_size)
        exps = _names("alpha", model_size) + _names("beta", dataset_size)
        return LawInstance(
            law=_BoundedRationalPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "I", *weights, *exps),
            # Two offsets and no amplitude: the law is linear in its two limits. The ``A_i`` sit
            # inside the ratio, so they are nonlinear weights rather than amplitudes.
            param_kinds=(OFFSET, OFFSET, *([SCALE] * len(predictors)), *([EXPONENT] * len(predictors))),
            exponent_predictor=tuple(range(len(predictors))),
        )

    @staticmethod
    def nonlinear_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
        """A weight and an exponent per predictor.

        A weight of one puts the centre of the design halfway between the two limits, so the usual ratio grid
        -- centred on one -- starts the search where the bend is observable.
        """
        k = len(law.predictors)
        return (*([SCALE_SPEC] * k), *([EXPONENT_SPEC] * k))

    @staticmethod
    def min_levels(law: LawInstance) -> tuple[int, ...]:
        """Three levels each on a grid; four when a single predictor has to identify all four parameters."""
        return (4,) if len(law.predictors) == 1 else (3,) * len(law.predictors)

    @staticmethod
    def _weight(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each predictor's term ``A_i * x_i**-alpha_i`` and their sum ``g``."""
        k = len(law.predictors)
        terms = nonlinear[..., None, :k] * np.exp(-nonlinear[..., None, k:] * log_x)
        return terms, terms.sum(axis=-1)

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1 / (1 + g), g / (1 + g)]``: the weights on ``E`` and on ``I``."""
        _, g = _BoundedRationalPower._weight(law, nonlinear, log_x)
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0 / (1.0 + g)
        out[:, 1] = g / (1.0 + g)
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many nonlinear vectors at once, shape ``(n_draws, n_points, 2)``."""
        _, g = _BoundedRationalPower._weight(law, nonlinear, log_x)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0 / (1.0 + g)
        out[:, :, 1] = g / (1.0 + g)
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """``d f / d g = (I - E) / (1 + g)**2``, chained through each term of ``g``."""
        k = len(law.predictors)
        terms, g = _BoundedRationalPower._weight(law, nonlinear, log_x)
        slope = ((linear[1] - linear[0]) / (1.0 + g) ** 2)[:, None]
        grad = np.empty((log_x.shape[0], 2 * k), dtype=float)
        grad[:, :k] = slope * np.exp(-nonlinear[None, k:] * log_x)
        grad[:, k:] = -slope * terms * log_x
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """The limits are scale-free; each weight absorbs ``reference**alpha`` like an amplitude."""
        values = _values(law, params)
        k = len(law.predictors)
        exponents = params[law.n_linear + k :]
        for i, name in enumerate(law.param_names[law.n_linear : law.n_linear + k]):
            values[name] *= float(reference[i] ** exponents[i])
        return values


class _BrokenPower:
    """``f(x) = E + A * x ** -c0 * (1 + (x / d) ** (1 / f)) ** (-c1 * f)``, for a single predictor.

    The one-break **Broken Neural Scaling Law** of Caballero et al. (2022), *Broken Neural Scaling
    Laws* -- their functional form with ``n = 1`` break. Every other law here has one slope on
    log-log axes. This one has two, joined smoothly: below the break scale ``d`` the excess loss
    falls as ``x ** -c0``, above it as ``x ** -(c0 + c1)``, and ``f`` sets how many e-folds of scale
    the changeover takes. ``c1`` is the *change* in slope and may have either sign, so the curve can
    steepen (a delayed take-off), flatten (early saturation), or -- when ``c1 < -c0`` -- turn back up,
    which is what a single descent of double descent looks like along one axis.

    **Why only one predictor and one break.** The law spends four nonlinear parameters on one axis
    and has six in all, so that axis must be measured at six or more scales before the break is
    identified at all. A second break, or a break in each of two predictors, would need scale grids
    well beyond the experiments this package is for, and would fit them exactly long before it
    identified anything.

    **Where the break may be.** Only inside the range of scales that was run. A break below the
    smallest scale leaves just the upper slope visible, and one above the largest just the lower: in
    both cases ``c0`` and ``c1`` can no longer be told apart, and the fit would be an ordinary power
    law with two spare parameters. So ``d`` is confined to the observed range, and a break estimated
    *at* either end is reported as ``parameter_at_bound`` -- read that as "no break was observed".
    The changeover width is confined in the same spirit: no sharper than a hundredth of the observed
    range, which no grid could resolve, and no wider than half of it, beyond which there is no
    visible break either. At ``c1 = 0`` the law is exactly a single-predictor ``separable-power``.

    Because the break is placed from the data, an extrapolation simply continues the upper slope. It
    cannot anticipate a second regime change beyond the grid, which is the standing caveat of every
    extrapolation here made concrete.
    """

    name: ClassVar[str] = "broken-power"
    description: ClassVar[str] = (
        "E + A * x**-c0 * (1 + (x / d)**(1 / f))**(-c1 * f) (one smooth change of slope; single predictor)"
    )

    @staticmethod
    def build(model_size: tuple[str, ...], dataset_size: tuple[str, ...]) -> LawInstance:
        """Construct the :class:`LawInstance` for these predictors."""
        if len(model_size) + len(dataset_size) != 1:
            raise LawError(
                f"Law {_BrokenPower.name!r} needs exactly one predictor, got {len(model_size)} "
                f"model-size and {len(dataset_size)} dataset-size"
            )
        return LawInstance(
            law=_BrokenPower.name,
            model_size=model_size,
            dataset_size=dataset_size,
            param_names=("E", "A", "c0", "c1", "d", "f"),
            param_kinds=(OFFSET, AMPLITUDE, EXPONENT, SIGNED_EXPONENT, LOCATION, WIDTH),
            exponent_predictor=(0, 0),
        )

    @staticmethod
    def nonlinear_specs(law: LawInstance, log_x: np.ndarray | None) -> tuple[NonlinearSpec, ...]:
        """The lower slope, the change of slope, and a break placed and sized from the observed range.

        The seed grid is spent where this law's cost surface is rugged. Moving the break by a fraction of the
        observed range changes which points sit on which slope, so basins in the break's position and width
        are narrow; the two slopes are smooth by comparison. The break therefore starts at each sixth of the
        observed log range and the changeover at a fortieth, a tenth and a quarter of it, while the lower
        slope gets three starts rather than the usual six. With the break at quarters and two widths, five of
        twelve simulated turnovers on an eight-decade sweep converged to a visibly worse fit; with this grid
        none did.
        """
        low, high = _log_range(log_x, 0)
        span = max(high - low, np.finfo(float).eps)
        slope = NonlinearSpec(0.0, MAX_EXPONENT, (0.05, 0.3, 1.0))
        location = NonlinearSpec(
            float(np.exp(low)),
            float(np.exp(high)),
            tuple(float(np.exp(low + q * span / 6.0)) for q in range(1, 6)),
            resolution=float(np.exp(low)),
        )
        width = NonlinearSpec(span / 100.0, span / 2.0, (span / 40.0, span / 10.0, span / 4.0))
        return (slope, SLOPE_CHANGE_SPEC, location, width)

    @staticmethod
    def min_levels(law: LawInstance) -> tuple[int, ...]:
        """Six parameters on one axis: nothing is identified below six scales."""
        return (law.n_params,)

    @staticmethod
    def _log_basis(nonlinear: np.ndarray, log_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log of the power-law factor, with the standardized distance past the break and its softplus.

        Written in logs throughout: ``(1 + (x / d) ** (1 / f)) ** (-c1 * f)`` is
        ``exp(-c1 * f * softplus(z))`` with ``z = log(x / d) / f``, which neither overflows for a sharp
        break far from ``d`` nor loses the small-``x`` limit.
        """
        c0, c1, d, f = (nonlinear[..., i, None] for i in range(4))
        z = (log_x[:, 0] - np.log(d)) / f
        softplus = np.logaddexp(0.0, z)
        return -c0 * log_x[:, 0] - c1 * f * softplus, z, softplus

    @staticmethod
    def design(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Columns ``[1, x**-c0 * (1 + (x / d)**(1 / f))**(-c1 * f)]``."""
        log_basis, _, _ = _BrokenPower._log_basis(nonlinear, log_x)
        out = np.empty((log_x.shape[0], 2), dtype=float)
        out[:, 0] = 1.0
        out[:, 1] = np.exp(log_basis)
        return out

    @staticmethod
    def many_designs(law: LawInstance, nonlinear: np.ndarray, log_x: np.ndarray) -> np.ndarray:
        """Design matrices for many nonlinear vectors at once, shape ``(n_draws, n_points, 2)``."""
        log_basis, _, _ = _BrokenPower._log_basis(nonlinear, log_x)
        out = np.empty((nonlinear.shape[0], log_x.shape[0], 2), dtype=float)
        out[:, :, 0] = 1.0
        out[:, :, 1] = np.exp(log_basis)
        return out

    @staticmethod
    def nonlinear_grad(
        law: LawInstance, linear: np.ndarray, nonlinear: np.ndarray, design: np.ndarray, log_x: np.ndarray
    ) -> np.ndarray:
        """Derivatives of ``A * exp(-c0 log x - c1 f softplus(z))`` with ``z = log(x / d) / f``."""
        _, c1, d, f = nonlinear
        _, z, softplus = _BrokenPower._log_basis(nonlinear, log_x)
        sigmoid = np.exp(z - softplus)
        value = linear[1] * design[:, 1]
        grad = np.empty((log_x.shape[0], 4), dtype=float)
        grad[:, 0] = -value * log_x[:, 0]
        grad[:, 1] = -value * f * softplus
        grad[:, 2] = value * c1 * sigmoid / d
        grad[:, 3] = -value * c1 * (softplus - sigmoid * z)
        return grad

    @staticmethod
    def raw_scale(law: LawInstance, params: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        """The amplitude absorbs ``reference**c0``; the break is a scale, so it is simply rescaled."""
        values = _values(law, params)
        values["A"] *= float(reference[0] ** params[2])
        values["d"] *= float(reference[0])
        return values


_LAWS: dict[str, Any] = {
    law.name: law
    for law in (
        _SeparablePower,
        _MultiplicativePower,
        _ShiftedSeparablePower,
        _KaplanJointPower,
        _SkalingJointPower,
        _BoundedRationalPower,
        _BrokenPower,
    )
}
_DESIGNS = {name: law.design for name, law in _LAWS.items()}
_NONLINEAR_GRADS = {name: law.nonlinear_grad for name, law in _LAWS.items()}
_MANY_DESIGNS = {name: law.many_designs for name, law in _LAWS.items()}
_RAW_SCALE = {name: law.raw_scale for name, law in _LAWS.items()}
_SPECS = {name: law.nonlinear_specs for name, law in _LAWS.items()}
_MIN_LEVELS = {name: law.min_levels for name, law in _LAWS.items()}

#: Name of the law used when the user does not choose one.
DEFAULT_LAW = _SeparablePower.name


def available_laws() -> dict[str, str]:
    """Return a mapping from law name to a one-line description.

    Examples:
        >>> for name, description in available_laws().items():
        ...     print(f"{name}: {description}")
        separable-power: E + sum_i A_i * x_i**-alpha_i (one additive power term per predictor)
        multiplicative-power: E + A * prod_i x_i**-alpha_i (one joint multiplicative power term)
        shifted-separable-power: E + sum_i A_i * (x_i + s_i)**-alpha_i (power terms with a finite
        small-scale limit)
        kaplan-joint-power: E + A * (N**-rho + R / D)**beta (Kaplan et al. joint model/data law)
        skaling-joint-power: E + A * (N**-alpha + R * D**-beta)**k (Videau et al. Skaling law: coupled
        model/data terms)
        bounded-rational-power: (E + I * g) / (1 + g), g = sum_i A_i * x_i**-alpha_i (plateau I at small
        scale, floor E at large)
        broken-power: E + A * x**-c0 * (1 + (x / d)**(1 / f))**(-c1 * f) (one smooth change of slope;
        single predictor)
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
        LawError: If the law name is unknown, no predictors were supplied, or the law cannot be
            built over this number of predictors.

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
        multiplicative-power, shifted-separable-power, kaplan-joint-power, skaling-joint-power,
        bounded-rational-power, broken-power

        Some laws are defined only for a particular set of predictors, and say so:

        >>> build_law("kaplan-joint-power", ["model_size__n"], [])
        Traceback (most recent call last):
        ...
        simple_scaling_laws.laws.LawError: Law 'kaplan-joint-power' needs exactly one model-size and one
        dataset-size predictor, got 1 and 0
    """
    if law not in _LAWS:
        raise LawError(f"Unknown law {law!r}; available: {', '.join(_LAWS)}")
    model_size, dataset_size = tuple(model_size), tuple(dataset_size)
    if not model_size and not dataset_size:
        raise LawError(f"Law {law!r} needs at least one predictor column")
    instance = _LAWS[law].build(model_size, dataset_size)
    duplicated = sorted({name for name in instance.param_names if instance.param_names.count(name) > 1})
    if duplicated:
        raise LawError(
            f"Law {law!r} over {list(instance.predictors)} would give two parameters the same name "
            f"{duplicated}; rename a predictor column so their suffixes differ"
        )
    return instance
