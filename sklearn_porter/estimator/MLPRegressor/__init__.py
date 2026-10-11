# scikit-learn
from sklearn.neural_network import \
    MLPRegressor as MLPRegressorClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.MLPClassifier import MLPClassifier


class MLPRegressor(MLPClassifier, EstimatorBase):
    """Extract model data and port a MLPRegressor regressor."""

    SKLEARN_URL = 'sklearn.neural_network.MLPRegressor.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.JS: {
            enum.Template.ATTACHED: {
                enum.Method.PREDICT,
            },
            enum.Template.EXPORTED: {
                enum.Method.PREDICT,
            },
        },
    }

    estimator = None  # type: MLPRegressorClass

    def __init__(self, estimator: MLPRegressorClass):

        try:
            estimator.coefs_
        except AttributeError:
            estimator_name = estimator.__class__.__qualname__
            raise exception.NotFittedEstimatorError(estimator_name)

        super().__init__(estimator)
