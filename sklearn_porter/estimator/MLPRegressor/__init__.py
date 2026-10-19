# scikit-learn
from sklearn.neural_network import MLPRegressor as MLPRegressorClass

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
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.CS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {
                enum.Method.PREDICT,
            },
            enum.Template.EXPORTED: {
                enum.Method.PREDICT,
            },
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
    }

    def __init__(self, estimator: MLPRegressorClass):

        try:
            estimator.coefs_
        except AttributeError:
            estimator_name = estimator.__class__.__qualname__
            raise exception.NotFittedEstimatorError(estimator_name) from None

        # The ported regressors support the output activations `identity`
        # and `exp` (with `loss='poisson'`):
        if estimator.out_activation_ not in ('identity', 'exp'):
            msg = (
                f'The output activation `{estimator.out_activation_}` is not '
                'supported, only `identity` and `exp`.'
            )
            raise exception.NotSupportedYetError(msg)

        super().__init__(estimator)
