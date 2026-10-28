from collections.abc import Callable

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.preprocessing import KBinsDiscretizer as KBinsDiscretizerClass

# sklearn-porter
from sklearn_porter import enums as enum
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

        # scikit-learn counts the inner edges of the bins of each feature
        # which are lower than or equal to the feature (`np.searchsorted`
        # with `side='right'`). The inner edges of all features are
        # concatenated, the edges of feature `i` are at the positions
        # `offsets[i]` to `offsets[i + 1] - 1`:
        edges = [np.asarray(e, dtype=np.float64)[1:-1] for e in est.bin_edges_]
        offsets = np.cumsum([0] + [len(e) for e in edges])
        self.feature_edges = [e.tolist() for e in edges]

        # With `encode='onehot'` (a sparse matrix) or `encode='onehot-dense'`
        # each bin is a transformed feature, the bins of feature `i` start at
        # the position `offsets[i] + i`. A bin is the interval between two
        # edges (the first and the last bin are unbounded):
        self.onehot = est.encode != 'ordinal'
        self.columns = []
        for i, e in enumerate(self.feature_edges):
            bounds = [None, *e, None]
            self.columns += [
                (i, bounds[b], bounds[b + 1]) for b in range(len(e) + 1)
            ]
        n_features_out = len(self.columns) if self.onehot else n_features

        self._set_model_data(
            n_features,
            n_features_out=n_features_out,
            ints=('offsets',),
            edges=np.concatenate(edges),
            offsets=offsets,
        )

    def literals(self, converter: Callable[[object], str]) -> dict:
        def bound(value: float | None) -> str | None:
            return None if value is None else converter(value)

        return dict(
            feature_edges=[list(map(converter, e)) for e in self.feature_edges],
            onehot=self.onehot,
            column_inputs=[c[0] for c in self.columns],
            column_lower=[bound(c[1]) for c in self.columns],
            column_upper=[bound(c[2]) for c in self.columns],
        )
