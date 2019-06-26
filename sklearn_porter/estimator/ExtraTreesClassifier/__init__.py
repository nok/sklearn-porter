# -*- coding: utf-8 -*-

from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifierClass

from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.RandomForestClassifier import \
    RandomForestClassifier
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class ExtraTreesClassifier(RandomForestClassifier, EstimatorBase):
    """Extract model data and port an ExtraTreesClassifier classifier."""

    estimator = None  # type: ExtraTreesClassifierClass

    def __init__(self, estimator: ExtraTreesClassifierClass):
        super().__init__(estimator)
