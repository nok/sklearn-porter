# scikit-learn
from sklearn.svm import OneClassSVM as OneClassSVMClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.SVR import SVR


class OneClassSVM(SVR, EstimatorBase):
    """Extract model data and port a OneClassSVM outlier detector."""

    SKLEARN_URL = 'sklearn.svm.OneClassSVM.html'

    # Predict 1 for inliers and -1 for outliers like scikit-learn:
    IS_OUTLIER_DETECTOR = True

    def __init__(self, estimator: OneClassSVMClass):
        super().__init__(estimator)
