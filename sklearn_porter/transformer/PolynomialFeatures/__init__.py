from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import PolynomialFeatures as PolynomialFeaturesClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class PolynomialFeatures(TransformerBase, EstimatorApiABC):
    """Extract model data and port a PolynomialFeatures transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.PolynomialFeatures.html'

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

    def __init__(self, transformer: PolynomialFeaturesClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # Each transformed feature is the product of its factors (the
        # positions of the features), the factors of all transformed features
        # are concatenated, the factors of the transformed feature `i` are at
        # the positions `offsets[i]` to `offsets[i + 1] - 1`:
        self.column_factors = _get_factors(est, n_features)
        offsets = np.cumsum([0] + [len(f) for f in self.column_factors])
        self._set_model_data(
            n_features,
            n_features_out=len(self.column_factors),
            ints=('factors', 'offsets'),
            # A dummy factor if there is only the bias, so no array is empty:
            factors=[f for fs in self.column_factors for f in fs] or [0],
            offsets=offsets,
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(column_factors=self.column_factors)


def _get_factors(est: PolynomialFeaturesClass, n_features: int) -> list:
    """
    Get the factors of each transformed feature.

    scikit-learn computes the terms of degree `d` by multiplying a term of
    degree `d - 1` with a feature (`PolynomialFeatures.transform` of dense
    data), so the factors are in the order of these multiplications and the
    ported transformer gets exactly the same products.

    Parameters
    ----------
    est : PolynomialFeatures
        The transformer of scikit-learn.
    n_features : int
        The number of features.

    Returns
    -------
    The positions of the features of each transformed feature (an empty
    list for the bias).
    """
    # The degree is the maximal degree or a tuple of the minimal and the
    # maximal degree:
    degree = est.degree
    if isinstance(degree, tuple):
        min_degree, max_degree = degree
    else:
        min_degree, max_degree = 0, degree
    columns = [[]] if est.include_bias else []
    if max_degree > 0:
        start = len(columns)
        columns += [[i] for i in range(n_features)]
        index = [*range(start, start + n_features), start + n_features]
        current = start + n_features
        for _ in range(2, max_degree + 1):
            new_index, end = [], index[-1]
            for i in range(n_features):
                begin = index[i]
                new_index.append(current)
                if est.interaction_only:
                    begin += index[i + 1] - index[i]
                if end <= begin:
                    break
                columns += [columns[c] + [i] for c in range(begin, end)]
                current += end - begin
            new_index.append(current)
            index = new_index

    # Without the terms of a degree lower than `min_degree` (but the bias):
    n_out = est.n_output_features_
    if min_degree > 1:
        if est.include_bias:
            columns = columns[:1] + columns[len(columns) - n_out + 1 :]
        else:
            columns = columns[len(columns) - n_out :]

    # The powers of the features must match the powers of scikit-learn:
    powers = np.zeros((len(columns), n_features), dtype=int)
    for row, factors in zip(powers, columns, strict=True):
        np.add.at(row, factors, 1)
    if len(columns) != n_out or not np.array_equal(powers, est.powers_):
        msg = (
            f'The transformer `{type(est).__name__}` has unexpected powers '
            'of the features.'
        )
        raise exception.NotSupportedYetError(msg)
    return columns
