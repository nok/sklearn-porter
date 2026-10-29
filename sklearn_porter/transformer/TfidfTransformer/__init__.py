from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.feature_extraction.text import (
    TfidfTransformer as TfidfTransformerClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class TfidfTransformer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a TfidfTransformer transformer."""

    SKLEARN_URL = 'sklearn.feature_extraction.text.TfidfTransformer.html'

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

    def __init__(self, transformer: TfidfTransformerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        self.norm = est.norm
        self.sublinear_tf = est.sublinear_tf
        # scikit-learn weights the features with `idf_`, if the fit created
        # it (`use_idf=True`), independent of the current parameter
        # `use_idf`. Multiplying by one doesn't change the features:
        idf = getattr(est, 'idf_', None)
        if idf is None:
            idf = np.ones(n_features)
        self._set_model_data(n_features, idf=idf)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(norm=self.norm, sublinear_tf=self.sublinear_tf)
