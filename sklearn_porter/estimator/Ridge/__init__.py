# scikit-learn
from sklearn.linear_model import Ridge as RidgeClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class Ridge(LinearRegression, EstimatorBase):
    """Extract model data and port a Ridge regressor."""

    SKLEARN_URL = 'sklearn.linear_model.Ridge.html'

    def __init__(self, estimator: RidgeClass):
        super().__init__(estimator)
