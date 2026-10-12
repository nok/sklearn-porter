# scikit-learn
from sklearn.linear_model import ElasticNet as ElasticNetClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class ElasticNet(LinearRegression, EstimatorBase):
    """Extract model data and port a ElasticNet regressor."""

    SKLEARN_URL = 'sklearn.linear_model.ElasticNet.html'

    def __init__(self, estimator: ElasticNetClass):
        super().__init__(estimator)
