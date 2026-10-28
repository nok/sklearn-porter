from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import Normalizer as NormalizerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class Normalizer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a Normalizer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.Normalizer.html'

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

    def __init__(self, transformer: NormalizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # The norms `l1` and `l2` are sums, which NumPy computes in another
        # order (pairwise summation), so only the maximum is identical:
        if est.norm != 'max':
            msg = (
                f'The transformer `{self.estimator_name}` supports only '
                "`norm='max'`."
            )
            raise exception.NotSupportedYetError(msg)
        self._set_model_data(n_features)

    def literals(self, converter: Callable[[object], str]) -> dict:
        # scikit-learn divides by one instead of a norm close to zero, see
        # `sklearn.preprocessing._data._handle_zeros_in_scale`:
        return dict(min_norm=converter(10 * np.finfo(np.float64).eps))
