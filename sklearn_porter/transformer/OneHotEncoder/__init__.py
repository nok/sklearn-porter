import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import OneHotEncoder as OneHotEncoderClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class OneHotEncoder(TransformerBase, EstimatorApiABC):
    """Extract model data and port a OneHotEncoder transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.OneHotEncoder.html'

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

    def __init__(self, transformer: OneHotEncoderClass):
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

        # Each category of each feature is a transformed feature (without the
        # dropped category of a feature, see `drop`), which is 1 if the
        # feature (`inputs`) is the category and 0 otherwise. Only numbers
        # are supported as categories, a missing value (`NaN`, the last
        # category of a feature) is marked by `missing`. Unknown categories
        # and the dropped categories match no transformed feature:
        drop_idx = getattr(est, 'drop_idx_', None)
        inputs, categories, missing = [], [], []
        for i, cats in enumerate(est.categories_):
            if np.asarray(cats).dtype.kind not in 'iuf':
                msg = (
                    f'The transformer `{self.estimator_name}` supports only '
                    'numbers as categories.'
                )
                raise exception.NotSupportedYetError(msg)
            drop = None if drop_idx is None else drop_idx[i]
            for j, category in enumerate(np.asarray(cats, dtype=np.float64)):
                if drop is not None and j == drop:
                    continue
                inputs.append(i)
                missing.append(int(np.isnan(category)))
                categories.append(0.0 if np.isnan(category) else category)
        if not inputs:
            msg = (
                f'The transformer `{self.estimator_name}` has no transformed '
                'features.'
            )
            raise exception.NotSupportedYetError(msg)

        self._set_model_data(
            n_features,
            n_features_out=len(inputs),
            ints=('inputs', 'missing'),
            inputs=inputs,
            categories=categories,
            missing=missing,
        )
