import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import RobustScaler as RobustScalerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class RobustScaler(TransformerBase, EstimatorApiABC):
    """Extract model data and port a RobustScaler transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.RobustScaler.html'

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

    def __init__(self, transformer: RobustScalerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # scikit-learn subtracts the center only with `with_centering=True`
        # and divides by the scale only with `with_scaling=True` (`center_`
        # or `scale_` are `None` otherwise). Subtracting zero and dividing by
        # one doesn't change any value (neither `-0.0` nor `NaN`), so the
        # ported transformer always uses both:
        center = est.center_ if est.with_centering else np.zeros(n_features)
        scale = est.scale_ if est.with_scaling else np.ones(n_features)
        self._set_model_data(n_features, center=center, scale=scale)
