from typing import Any

from loguru import logger as L

# scikit-learn
from sklearn.base import BaseEstimator

# sklearn-porter
from sklearn_porter.transformer.ColumnTransformer import ColumnTransformer


class FeatureUnion(ColumnTransformer):
    """
    Extract the parts and port a FeatureUnion transformer.

    Each part transforms all features, the transformed features of all
    parts are concatenated (and multiplied by the weights of the parts),
    see `ColumnTransformer`.
    """

    SKLEARN_URL = 'sklearn.pipeline.FeatureUnion.html'

    def _get_parts(self, transformer: BaseEstimator, n_features: int) -> list:
        L.debug(f'Get the parts of `{self.estimator_name}`.')
        est: Any = transformer  # alias (with attributes of scikit-learn)
        weights = est.transformer_weights or {}
        columns = list(range(n_features))
        parts, offset = [], 0
        for name, trans in est.transformer_list:
            if isinstance(trans, str):  # `'drop'`
                continue
            part = self._create_part(name, trans, columns, weights.get(name))
            part['offset'] = offset
            offset += part['n_out']
            parts.append(part)
        return parts
