# -*- coding: utf-8 -*-

from sklearn.svm.classes import NuSVC as NuSVCClass

from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.SVC import SVC
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class NuSVC(SVC, EstimatorBase):
    """
    Extract model data and port a NuSVC classifier.

    See also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    """
    estimator = None  # type: NuSVCClass

    def __init__(self, estimator: NuSVCClass):
        super().__init__(estimator)
