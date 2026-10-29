from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import (
    QuantileTransformer as QuantileTransformerClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class QuantileTransformer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a QuantileTransformer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.QuantileTransformer.html'

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

    def __init__(self, transformer: QuantileTransformerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # The normal distribution needs the quantile function of SciPy
        # (`scipy.stats.norm.ppf`), which can't be ported exactly:
        if est.output_distribution != 'uniform':
            msg = (
                f'The transformer `{self.estimator_name}` supports only '
                "`output_distribution='uniform'`."
            )
            raise exception.NotSupportedYetError(msg)

        # scikit-learn interpolates each feature between its quantiles
        # (`quantiles_[:, i]`) and the references (`np.interp` in both
        # directions, see `QuantileTransformer._transform_col`). The
        # quantiles of all features are concatenated, the quantiles of
        # feature `i` are at the positions `i * n_quantiles` to
        # `(i + 1) * n_quantiles - 1`:
        quantiles = np.asarray(est.quantiles_, dtype=np.float64)
        self.n_quantiles = quantiles.shape[0]
        self.feature_quantiles = quantiles.T.tolist()
        self._set_model_data(
            n_features,
            quantiles=quantiles.T,
            references=est.references_,
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            n_quantiles=self.n_quantiles,
            feature_quantiles=[
                list(map(converter, q)) for q in self.feature_quantiles
            ],
        )
