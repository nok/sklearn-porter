from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import MinMaxScaler as MinMaxScalerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class MinMaxScaler(TransformerBase, EstimatorApiABC):
    """Extract model data and port a MinMaxScaler transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.MinMaxScaler.html'

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

    def __init__(self, transformer: MinMaxScalerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # scikit-learn multiplies by the scale and adds the minimum
        # (`X * scale_ + min_`), so the ported transformer does the same in
        # the same order:
        self._set_model_data(n_features, scale=est.scale_, min=est.min_)

        # The bounds of `clip=True` are the parameter `feature_range`:
        self.clip = bool(est.clip)
        self.feature_range = tuple(map(float, est.feature_range))
        if self.clip and not np.isfinite(self.feature_range).all():
            msg = (
                'An infinite `feature_range` of the transformer '
                f'`{self.estimator_name}` is not supported yet.'
            )
            raise exception.NotSupportedYetError(msg)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            clip=self.clip,
            clip_min=converter(self.feature_range[0]),
            clip_max=converter(self.feature_range[1]),
        )
