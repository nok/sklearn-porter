# scikit-learn
from sklearn.linear_model import RidgeClassifierCV as RidgeClassifierCVClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.RidgeClassifier import RidgeClassifier


class RidgeClassifierCV(RidgeClassifier, EstimatorBase):
    """Extract model data and port a RidgeClassifierCV classifier."""

    SKLEARN_URL = 'sklearn.linear_model.RidgeClassifierCV.html'

    def __init__(self, estimator: RidgeClassifierCVClass):
        super().__init__(estimator)
