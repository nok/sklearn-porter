# scikit-learn
from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesRegressorClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.RandomForestRegressor import (
    RandomForestRegressor,
)


class ExtraTreesRegressor(RandomForestRegressor, EstimatorBase):
    """Extract model data and port an ExtraTreesRegressor regressor."""

    SKLEARN_URL = 'sklearn.ensemble.ExtraTreesRegressor.html'

    estimator = None  # type: ExtraTreesRegressorClass

    def __init__(self, estimator: ExtraTreesRegressorClass):
        super().__init__(estimator)
