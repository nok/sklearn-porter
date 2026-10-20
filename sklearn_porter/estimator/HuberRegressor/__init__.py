# scikit-learn
from sklearn.linear_model import HuberRegressor as HuberRegressorClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class HuberRegressor(LinearRegression, EstimatorBase):
    """Extract model data and port a HuberRegressor regressor."""

    SKLEARN_URL = 'sklearn.linear_model.HuberRegressor.html'

    def __init__(self, estimator: HuberRegressorClass):
        super().__init__(estimator)
