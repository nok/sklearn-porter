from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import OrdinalEncoder as OrdinalEncoderClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class OrdinalEncoder(TransformerBase, EstimatorApiABC):
    """Extract model data and port a OrdinalEncoder transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.OrdinalEncoder.html'

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

    def __init__(self, transformer: OrdinalEncoderClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # Infrequent categories are grouped to one category:
        infrequent = getattr(est, 'infrequent_categories_', None) or []
        if any(c is not None for c in infrequent):
            msg = (
                f'The transformer `{self.estimator_name}` with infrequent '
                'categories (`max_categories` or `min_frequency`) is not '
                'supported yet.'
            )
            raise exception.NotSupportedYetError(msg)

        # Only numbers are supported as categories, a missing value (`NaN`)
        # is the last category of a feature:
        categories, missing = [], []
        for cats in est.categories_:
            if np.asarray(cats).dtype.kind not in 'iuf':
                msg = (
                    f'The transformer `{self.estimator_name}` supports only '
                    'numbers as categories.'
                )
                raise exception.NotSupportedYetError(msg)
            cats = np.asarray(cats, dtype=np.float64)
            missing.append(int(np.isnan(cats).any()))
            categories.append(cats[~np.isnan(cats)])

        # The categories of all features are concatenated, the categories of
        # feature `i` are at the positions `offsets[i]` to
        # `offsets[i + 1] - 1`:
        offsets = np.cumsum([0] + [len(c) for c in categories])
        self.feature_categories = [c.tolist() for c in categories]
        self.feature_missing = missing
        self._set_model_data(
            n_features,
            ints=('offsets', 'missing'),
            categories=np.concatenate(categories),
            offsets=offsets,
            missing=missing,
        )

        # The value of unknown categories (scikit-learn raises an error with
        # `handle_unknown='error'`, the ported transformer returns `NaN`):
        unknown_value = est.unknown_value
        if est.handle_unknown != 'use_encoded_value' or unknown_value is None:
            unknown_value = np.nan
        self.unknown_value = float(unknown_value)
        self.encoded_missing_value = float(est.encoded_missing_value)

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            feature_categories=[
                list(map(converter, c)) for c in self.feature_categories
            ],
            feature_missing=self.feature_missing,
            unknown_value=converter(self.unknown_value),
            encoded_missing_value=converter(self.encoded_missing_value),
        )
