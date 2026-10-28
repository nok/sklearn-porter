from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import MaxAbsScaler as MaxAbsScalerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class MaxAbsScaler(TransformerBase, EstimatorApiABC):
    """Extract model data and port a MaxAbsScaler transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.MaxAbsScaler.html'

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

    def __init__(self, transformer: MaxAbsScalerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)
        self._set_model_data(n_features, scale=est.scale_)
