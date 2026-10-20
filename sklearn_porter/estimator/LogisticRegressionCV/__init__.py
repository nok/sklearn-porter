# scikit-learn
from sklearn.linear_model import (
    LogisticRegressionCV as LogisticRegressionCVClass,
)

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LogisticRegression import LogisticRegression


class LogisticRegressionCV(LogisticRegression, EstimatorBase):
    """Extract model data and port a LogisticRegressionCV classifier."""

    SKLEARN_URL = 'sklearn.linear_model.LogisticRegressionCV.html'

    def __init__(self, estimator: LogisticRegressionCVClass):
        super().__init__(estimator)
