# scikit-learn
from sklearn.linear_model import RidgeCV as RidgeCVClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class RidgeCV(LinearRegression, EstimatorBase):
    """Extract model data and port a RidgeCV regressor."""

    SKLEARN_URL = 'sklearn.linear_model.RidgeCV.html'

    def __init__(self, estimator: RidgeCVClass):
        super().__init__(estimator)
