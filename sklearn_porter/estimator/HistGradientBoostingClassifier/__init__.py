from logging import getLogger

# scikit-learn
from sklearn.ensemble import (
    HistGradientBoostingClassifier as HistGradientBoostingClassifierClass,
)

# sklearn-porter
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.GradientBoostingClassifier import (
    GradientBoostingClassifier,
)
from sklearn_porter.estimator.HistGradientBoostingRegressor import (
    HistGradientBoostingRegressor,
    get_hist_trees,
)
from sklearn_porter.utils import get_labels

L = getLogger(__name__)


class HistGradientBoostingClassifier(
    GradientBoostingClassifier, HistGradientBoostingRegressor, EstimatorBase
):
    """Extract model data and port a HistGradientBoostingClassifier classifier."""

    SKLEARN_URL = 'sklearn.ensemble.HistGradientBoostingClassifier.html'

    def __init__(self, estimator: HistGradientBoostingClassifierClass):
        EstimatorBase.__init__(self, estimator)
        est = estimator  # alias

        # Is the estimator fitted?
        if not getattr(est, '_predictors', None):
            raise exception.NotFittedEstimatorError(self.estimator_name)

        # Binary classifiers have one tree per iteration and multiclass
        # classifiers one tree per class:
        trees, self._combined_trees, inits = get_hist_trees(est)
        labels, label_kind = get_labels(est)
        self.meta_info = dict(
            n_estimators=len(trees),
            n_features=est.n_features_in_,
            n_outputs=1,  # the number of values of each leaf
            n_classes=len(est.classes_),
            n_trees_per_stage=len(inits),
            is_binary=len(inits) == 1,
            is_exponential=False,
            # A binary classifier predicts the negative class if the raw
            # prediction is 0 (`raw > 0` like `argmax` of the probabilities):
            is_positive_at_zero=False,
            labels=labels,
            label_kind=label_kind,
        )
        L.debug('Meta info: %s', self.meta_info)

        self.model_data = dict(estimators=trees, inits=inits)
        L.debug('Model data: %s', self.model_data)
