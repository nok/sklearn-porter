# -*- coding: utf-8 -*-

# scikit-learn
from sklearn.svm.classes import NuSVC as NuSVCClass

# sklearn-porter
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.SVC import SVC


class NuSVC(SVC, EstimatorBase):
    """Extract model data and port a NuSVC classifier."""

    estimator = None  # type: NuSVCClass

    def __init__(self, estimator: NuSVCClass):
        super().__init__(estimator)
