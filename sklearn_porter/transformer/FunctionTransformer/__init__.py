from collections.abc import Callable
from numbers import Real

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import (
    FunctionTransformer as FunctionTransformerClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase

# The functions which can be ported (`func=None` is the identity):
FUNCTIONS = {
    'identity': None,
    'abs': np.abs,
    'sqrt': np.sqrt,
    'square': np.square,
    'exp': np.exp,
    'expm1': np.expm1,
    'log': np.log,
    'log1p': np.log1p,
    'negative': np.negative,
    'reciprocal': np.reciprocal,
    'floor': np.floor,
    'ceil': np.ceil,
    'sign': np.sign,
    'clip': np.clip,
}


class FunctionTransformer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a FunctionTransformer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.FunctionTransformer.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.TRANSFORM

    SUPPORT = {
        language: {
            enum.Template.ATTACHED: {enum.Method.TRANSFORM},
            enum.Template.COMBINED: {enum.Method.TRANSFORM},
            enum.Template.EXPORTED: {enum.Method.TRANSFORM},
        }
        for language in enum.Language
    }

    def __init__(self, transformer: FunctionTransformerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # The functions of NumPy which can be ported (`None` is the identity):
        func = est.func
        name = next(
            (name for name, f in FUNCTIONS.items() if f is func),
            None,
        )
        if name is None:
            msg = (
                f'The function of the transformer `{self.estimator_name}` '
                'is not supported yet. Supported are `None` and the '
                'functions of NumPy: '
                + ', '.join(
                    f'`np.{name}`' for name in FUNCTIONS if name != 'identity'
                )
                + '.'
            )
            raise exception.NotSupportedYetError(msg)
        self.func = name
        self.clip_min, self.clip_max = None, None
        if name == 'clip':
            self.clip_min, self.clip_max = self._get_bounds(est, n_features)
        elif est.kw_args:
            msg = (
                f'The transformer `{self.estimator_name}` with `kw_args` '
                'is supported only for `np.clip`.'
            )
            raise exception.NotSupportedYetError(msg)
        self._set_model_data(n_features)

    def _get_bounds(
        self, est: FunctionTransformerClass, n_features: int
    ) -> tuple[float | None, float | None]:
        """Get the bounds of `np.clip` (numbers or `None`) from `kw_args`."""
        kw_args = dict(est.kw_args or {})
        bounds = (
            kw_args.pop('a_min', kw_args.pop('min', None)),
            kw_args.pop('a_max', kw_args.pop('max', None)),
        )
        is_valid = not kw_args and any(b is not None for b in bounds)
        for bound in bounds:
            if bound is not None and (
                not isinstance(bound, Real)
                or isinstance(bound, bool)
                or np.isnan(bound)
            ):
                is_valid = False
        if is_valid and None not in bounds and bounds[0] > bounds[1]:
            is_valid = False
        if is_valid:
            try:  # e.g. `np.clip` requires `a_min` and `a_max`
                est.transform(np.zeros((1, n_features)))
            except (TypeError, ValueError):
                is_valid = False
        if not is_valid:
            msg = (
                f'The transformer `{self.estimator_name}` supports `np.clip` '
                'only with numbers as bounds (`a_min` and `a_max` or `min` '
                'and/or `max` in `kw_args`), the lower bound must not be '
                'greater than the upper bound.'
            )
            raise exception.NotSupportedYetError(msg)
        lower, upper = bounds
        return (
            None if lower is None else float(lower),
            None if upper is None else float(upper),
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        def bound(value: float | None) -> str | None:
            return None if value is None else converter(value)

        return dict(
            func=self.func,
            clip_min=bound(self.clip_min),
            clip_max=bound(self.clip_max),
        )
