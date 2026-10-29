from typing import Any

# scikit-learn
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.transformer.SelectorMixin import SelectorMixin


class SelectFromModel(SelectorMixin):
    """Extract model data and port a SelectFromModel, see `SelectorMixin`."""

    SKLEARN_URL = 'sklearn.feature_selection.SelectFromModel.html'

    def _get_n_features(self, transformer: BaseEstimator) -> int:
        est: Any = transformer  # alias (with attributes of scikit-learn)
        # With `prefit=True` scikit-learn selects the features with the
        # importances of the fitted estimator, also without a fit of the
        # selector (which sets the number of features):
        if est.prefit and not hasattr(est, 'estimator_'):
            try:
                return len(est.get_support())
            except NotFittedError:
                raise exception.NotFittedEstimatorError(
                    self.estimator_name
                ) from None
        return super()._get_n_features(transformer)
