# scikit-learn
from sklearn.linear_model import Lasso as LassoClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class Lasso(LinearRegression, EstimatorBase):
    """Extract model data and port a Lasso regressor."""

    SKLEARN_URL = 'sklearn.linear_model.Lasso.html'

    def __init__(self, estimator: LassoClass):
        super().__init__(estimator)
