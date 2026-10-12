# scikit-learn
from sklearn.linear_model import RidgeClassifier as RidgeClassifierClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.LogisticRegression import LogisticRegression


class RidgeClassifier(LogisticRegression, EstimatorBase):
    """Extract model data and port a RidgeClassifier classifier."""

    SKLEARN_URL = 'sklearn.linear_model.RidgeClassifier.html'

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.COMBINED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
    }

    def __init__(self, estimator: RidgeClassifierClass):
        super().__init__(estimator)
