from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import KBinsDiscretizer as KBinsDiscretizerClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.transformer.TransformerBase import TransformerBase


class KBinsDiscretizer(TransformerBase, EstimatorApiABC):
    """Extract model data and port a KBinsDiscretizer transformer."""

    SKLEARN_URL = 'sklearn.preprocessing.KBinsDiscretizer.html'

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

    def __init__(self, transformer: KBinsDiscretizerClass):
        super().__init__(transformer)
        L.info(f'Create specific transformer `{self.estimator_name}`.')
        est = transformer  # alias
        n_features = self._get_n_features(est)

        # One-hot encoded bins are additional features:
        if est.encode != 'ordinal':
            msg = (
                f'The transformer `{self.estimator_name}` supports only '
                "`encode='ordinal'`."
            )
            raise exception.NotSupportedYetError(msg)

        # scikit-learn counts the inner edges of the bins of each feature
        # which are lower than or equal to the feature (`np.searchsorted`
        # with `side='right'`). The inner edges of all features are
        # concatenated, the edges of feature `i` are at the positions
        # `offsets[i]` to `offsets[i + 1] - 1`:
        edges = [np.asarray(e, dtype=np.float64)[1:-1] for e in est.bin_edges_]
        offsets = np.cumsum([0] + [len(e) for e in edges])
        self.feature_edges = [e.tolist() for e in edges]
        self._set_model_data(
            n_features,
            ints=('offsets',),
            edges=np.concatenate(edges),
            offsets=offsets,
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        return dict(
            feature_edges=[list(map(converter, e)) for e in self.feature_edges]
        )
