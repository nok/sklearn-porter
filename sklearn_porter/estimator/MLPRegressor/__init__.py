# -*- coding: utf-8 -*-

from sklearn.neural_network.multilayer_perceptron \
    import MLPRegressor as MLPRegressorClass

from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.MLPClassifier import MLPClassifier
from sklearn_porter.enums import Method, Language, Template
from sklearn_porter.exceptions import NotFittedEstimatorError
from sklearn_porter.utils import get_logger


L = get_logger(__name__)


class MLPRegressor(MLPClassifier, EstimatorBase):
    """Extract model data and port a MLPRegressor regressor."""

    DEFAULT_LANGUAGE = Language.JS
    DEFAULT_METHOD = Method.PREDICT
    DEFAULT_TEMPLATE = Template.ATTACHED

    SUPPORT = {
        Language.JS: {Method.PREDICT: {Template.ATTACHED}},
    }

    estimator = None  # type: MLPRegressorClass

    def __init__(self, estimator: MLPRegressorClass):

        try:
            estimator.coefs_
        except AttributeError:
            estimator_name = estimator.__class__.__qualname__
            raise NotFittedEstimatorError(estimator_name)

        super().__init__(estimator)
