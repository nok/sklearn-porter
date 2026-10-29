from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class SelectorMixin(TransformerBase, EstimatorApiABC):
    """
    Extract model data and port a feature selector.

    All feature selectors of scikit-learn (subclasses of `SelectorMixin`,
    e.g. `SelectKBest` or `RFE`) keep the features of the mask of
    `get_support()` in `transform`, so the selectors share this class and
    its templates: each transformed feature is the selected feature
    `inputs[i]`. The subclasses are named after the selectors.

    Like scikit-learn the ported selectors reject samples with missing or
    infinite values (in any feature), unless the selector allows them
    (`allow_nan` of its tags, e.g. of a `VarianceThreshold` or of the
    estimator of a `SelectFromModel`).
    """

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

    def __init__(self, transformer: BaseEstimator):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est: Any = transformer  # alias (a selector of scikit-learn)
        n_features = self._get_n_features(est)

        # The mask of the selected features (e.g. of `SelectFromModel` with
        # the importances of the current estimator and the threshold):
        try:
            mask = np.asarray(est.get_support(), dtype=bool)
        except NotFittedError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None
        if mask.shape != (n_features,):
            msg = (
                f'The mask of the selected features of the transformer '
                f'`{self.estimator_name}` has not one value per feature.'
            )
            raise exception.NotSupportedYetError(msg)

        # scikit-learn returns no transformed features (with a warning), if
        # no feature is selected:
        inputs = np.flatnonzero(mask)
        if not inputs.size:
            msg = (
                f'The transformer `{self.estimator_name}` has no transformed '
                'features, because no feature is selected.'
            )
            raise exception.NotSupportedYetError(msg)
        self._set_model_data(
            n_features,
            n_features_out=inputs.size,
            ints=('inputs',),
            inputs=inputs,
        )

        # `transform` of scikit-learn checks that all features are finite,
        # unless the tags of the selector allow missing values:
        self.allow_nan = bool(get_tags(est).input_tags.allow_nan)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(allow_nan=self.allow_nan)
