from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import Normalizer as NormalizerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class Normalizer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a Normalizer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.Normalizer.html'

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

    def __init__(self, transformer: NormalizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        self.norm = est.norm
        self.norm_tree = _get_sum_tree(0, n_features)
        self._set_model_data(n_features)

    def literals(self, converter: Callable[[object], str]) -> dict:
        # scikit-learn divides by one instead of a norm close to zero, see
        # `sklearn.preprocessing._data._handle_zeros_in_scale`:
        return dict(
            norm=self.norm,
            norm_tree=self.norm_tree,
            min_norm=converter(10 * np.finfo(np.float64).eps),
        )


def _get_sum_tree(start: int, n: int) -> int | tuple:
    """
    Get the order of the sum of `n` values in NumPy (pairwise summation).

    scikit-learn computes the norm `l1` with `np.abs(X).sum(axis=1)`, which
    sums the values of each sample (a contiguous row) like
    `pairwise_sum` of NumPy: fewer than 8 values one after another, up to
    128 values with 8 partial sums and more values in two halves. The norm
    `l2` uses the same order, but scikit-learn computes the sum of the
    squares with `np.einsum`, whose order depends on the CPU.

    Parameters
    ----------
    start : int
        The position of the first value.
    n : int
        The number of values.

    Returns
    -------
    The position of a value or a tuple of two summands.
    """
    tree: int | tuple
    if n < 8:
        tree = start
        for i in range(start + 1, start + n):
            tree = (tree, i)
        return tree
    if n <= 128:
        sums: list[int | tuple] = list(range(start, start + 8))
        i = 8
        while i < n - n % 8:
            sums = [(s, start + i + j) for j, s in enumerate(sums)]
            i += 8
        tree = (
            ((sums[0], sums[1]), (sums[2], sums[3])),
            ((sums[4], sums[5]), (sums[6], sums[7])),
        )
        for k in range(i, n):
            tree = (tree, start + k)
        return tree
    half = n // 2
    half -= half % 8
    return (_get_sum_tree(start, half), _get_sum_tree(start + half, n - half))
