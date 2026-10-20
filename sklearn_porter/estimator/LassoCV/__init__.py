# scikit-learn
from sklearn.linear_model import LassoCV as LassoCVClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class LassoCV(LinearRegression, EstimatorBase):
    """Extract model data and port a LassoCV regressor."""

    SKLEARN_URL = 'sklearn.linear_model.LassoCV.html'

    def __init__(self, estimator: LassoCVClass):
        super().__init__(estimator)
