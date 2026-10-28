import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import Binarizer as BinarizerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class Binarizer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a Binarizer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.Binarizer.html'

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

    def __init__(self, transformer: BinarizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # The threshold of each feature (the same for all features):
        threshold = np.full(n_features, est.threshold, dtype=np.float64)
        self._set_model_data(n_features, threshold=threshold)
