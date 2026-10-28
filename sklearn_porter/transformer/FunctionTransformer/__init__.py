from collections.abc import Callable

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
        if est.kw_args:
            msg = (
                f'The transformer `{self.estimator_name}` with `kw_args` '
                'is not supported yet.'
            )
            raise exception.NotSupportedYetError(msg)
        self.func = name
        self._set_model_data(n_features)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(func=self.func)
