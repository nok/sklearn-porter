# scikit-learn
from sklearn.linear_model import ARDRegression as ARDRegressionClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class ARDRegression(LinearRegression, EstimatorBase):
    """Extract model data and port a ARDRegression regressor."""

    SKLEARN_URL = 'sklearn.linear_model.ARDRegression.html'

    def __init__(self, estimator: ARDRegressionClass):
        super().__init__(estimator)
