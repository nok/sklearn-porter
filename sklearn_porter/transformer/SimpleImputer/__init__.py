from collections.abc import Callable
from numbers import Real

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.impute import SimpleImputer as SimpleImputerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class SimpleImputer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a SimpleImputer transformer."""

    SKLEARN_URL = 'sklearn.impute.SimpleImputer.html'

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

    def __init__(self, transformer: SimpleImputerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # Missing values are `NaN` or a number:
        missing_values = est.missing_values
        is_number = isinstance(missing_values, Real)
        if not is_number or isinstance(missing_values, bool):
            msg = (
                f'The transformer `{self.estimator_name}` supports only '
                '`NaN` or a number as `missing_values`.'
            )
            raise exception.NotSupportedYetError(msg)
        self.missing_values = float(missing_values)

        # The statistics of the features (e.g. the means), which replace
        # the missing values:
        try:
            statistics = np.asarray(est.statistics_, dtype=np.float64)
        except (TypeError, ValueError):
            msg = (
                f'The transformer `{self.estimator_name}` supports only '
                'numbers as statistics (e.g. no strings of `fill_value`).'
            )
            raise exception.NotSupportedYetError(msg) from None

        # scikit-learn drops the features without any observed values in
        # the fit (their statistics are `NaN`), except with
        # `keep_empty_features=True`. With `add_indicator=True` the
        # indicators of the missing values of the features with missing
        # values in the fit (also of the dropped features) follow. Each
        # transformed feature is the imputed feature `inputs[i]` (with the
        # statistic `statistics[i]`) or its indicator (`indicators[i]`):
        kept = np.flatnonzero(~np.isnan(statistics))
        indicated = []
        if est.add_indicator:
            features = est.indicator_.features_  # ty: ignore[unresolved-attribute]
            indicated = np.asarray(features, dtype=int)
        inputs = [*kept.tolist(), *list(map(int, indicated))]
        if not inputs:
            msg = (
                f'The transformer `{self.estimator_name}` has no transformed '
                'features.'
            )
            raise exception.NotSupportedYetError(msg)
        self._set_model_data(
            n_features,
            n_features_out=len(inputs),
            ints=('inputs', 'indicators'),
            inputs=inputs,
            statistics=[*statistics[kept].tolist(), *[0.0] * len(indicated)],
            indicators=[0] * len(kept) + [1] * len(indicated),
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        is_nan = bool(np.isnan(self.missing_values))
        return dict(
            missing_values_nan=is_nan,
            missing_values=None if is_nan else converter(self.missing_values),
        )
