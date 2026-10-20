# scikit-learn
from sklearn.svm import NuSVR as NuSVRClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.SVR import SVR


class NuSVR(SVR, EstimatorBase):
    """Extract model data and port a NuSVR regressor."""

    SKLEARN_URL = 'sklearn.svm.NuSVR.html'

    def __init__(self, estimator: NuSVRClass):
        super().__init__(estimator)
