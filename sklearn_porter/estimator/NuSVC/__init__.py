# scikit-learn
from sklearn.svm import NuSVC as NuSVCClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.SVC import SVC


class NuSVC(SVC, EstimatorBase):
    """Extract model data and port a NuSVC classifier."""

    SKLEARN_URL = 'sklearn.svm.NuSVC.html'

    estimator = None  # type: NuSVCClass

    def __init__(self, estimator: NuSVCClass):
        super().__init__(estimator)
