# scikit-learn
from sklearn.linear_model import BayesianRidge as BayesianRidgeClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class BayesianRidge(LinearRegression, EstimatorBase):
    """Extract model data and port a BayesianRidge regressor."""

    SKLEARN_URL = 'sklearn.linear_model.BayesianRidge.html'

    def __init__(self, estimator: BayesianRidgeClass):
        super().__init__(estimator)
