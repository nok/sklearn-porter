# scikit-learn
from sklearn.linear_model import ElasticNetCV as ElasticNetCVClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class ElasticNetCV(LinearRegression, EstimatorBase):
    """Extract model data and port a ElasticNetCV regressor."""

    SKLEARN_URL = 'sklearn.linear_model.ElasticNetCV.html'

    def __init__(self, estimator: ElasticNetCVClass):
        super().__init__(estimator)
