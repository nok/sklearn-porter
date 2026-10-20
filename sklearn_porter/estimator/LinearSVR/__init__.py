# scikit-learn
from sklearn.svm import LinearSVR as LinearSVRClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LinearRegression import LinearRegression


class LinearSVR(LinearRegression, EstimatorBase):
    """Extract model data and port a LinearSVR regressor."""

    SKLEARN_URL = 'sklearn.svm.LinearSVR.html'

    def __init__(self, estimator: LinearSVRClass):
        super().__init__(estimator)
