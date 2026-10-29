from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import TargetEncoder as TargetEncoderClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class TargetEncoder(TransformerBase, EstimatorApiABC):
    """Extract model data and port a TargetEncoder transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.TargetEncoder.html'

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

    def __init__(self, transformer: TargetEncoderClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # Only numbers are supported as categories, a missing value (`NaN`)
        # is the last category of a feature:
        for cats in est.categories_:
            if np.asarray(cats).dtype.kind not in 'iuf':
                msg = (
                    f'The transformer `{self.estimator_name}` supports only '
                    'numbers as categories.'
                )
                raise exception.NotSupportedYetError(msg)

        # Each transformed feature is the encoding of the category of its
        # feature (`inputs`). With several classes (`target_type_` is
        # `multiclass`) there is one transformed feature per feature and
        # class (`encodings_` and the mean of the target per class). The
        # categories (without `NaN`) and their encodings of all transformed
        # features are concatenated, the ones of the transformed feature `i`
        # are at the positions `offsets[i]` to `offsets[i + 1] - 1`. Missing
        # values seen in the fit get their encoding (`nans`), unknown
        # categories (also unseen missing values) the mean of the target
        # (`unknowns`) like in scikit-learn:
        n_classes = 1
        target_means = np.atleast_1d(np.asarray(est.target_mean_, dtype=float))
        if est.target_type_ == 'multiclass':
            n_classes = len(est.classes_)  # ty: ignore[invalid-argument-type]
        inputs, categories, encodings, missing, nans, unknowns = (
            [] for _ in range(6)
        )
        self.columns = []
        for i, encoding in enumerate(est.encodings_):
            feature = i // n_classes
            cats = np.asarray(est.categories_[feature], dtype=np.float64)
            encoding = np.asarray(encoding, dtype=np.float64)
            is_nan = np.isnan(cats)
            unknown = float(target_means[i % n_classes])
            nan = float(encoding[is_nan][0]) if is_nan.any() else unknown
            inputs.append(feature)
            categories.append(cats[~is_nan])
            encodings.append(encoding[~is_nan])
            missing.append(int(is_nan.any()))
            nans.append(nan if is_nan.any() else 0.0)
            unknowns.append(unknown)
            self.columns.append(
                (
                    cats[~is_nan].tolist(),
                    encoding[~is_nan].tolist(),
                    nan,
                    unknown,
                )
            )
        if not any(len(c) for c in categories):
            msg = (
                f'The transformer `{self.estimator_name}` has no categories '
                'without missing values.'
            )
            raise exception.NotSupportedYetError(msg)
        offsets = np.cumsum([0] + [len(c) for c in categories])
        self._set_model_data(
            n_features,
            n_features_out=len(inputs),
            ints=('inputs', 'offsets', 'missing'),
            inputs=inputs,
            categories=np.concatenate(categories),
            encodings=np.concatenate(encodings),
            offsets=offsets,
            missing=missing,
            nans=nans,
            unknowns=unknowns,
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            column_pairs=[
                list(zip(map(converter, c), map(converter, e), strict=True))
                for c, e, _, _ in self.columns
            ],
            column_nan=[converter(nan) for _, _, nan, _ in self.columns],
            column_unknown=[converter(u) for _, _, _, u in self.columns],
        )
