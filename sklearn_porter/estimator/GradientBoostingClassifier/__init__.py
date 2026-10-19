from loguru import logger as L

# scikit-learn
from sklearn.ensemble import (
    GradientBoostingClassifier as GradientBoostingClassifierClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.GradientBoostingRegressor import (
    GradientBoostingRegressor,
    get_stages_data,
)
from sklearn_porter.utils import get_labels


class GradientBoostingClassifier(GradientBoostingRegressor, EstimatorBase):
    """Extract model data and port a GradientBoostingClassifier classifier."""

    SKLEARN_URL = 'sklearn.ensemble.GradientBoostingClassifier.html'

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.CS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: enum.ALL_METHODS,
            enum.Template.COMBINED: enum.ALL_METHODS,
            enum.Template.EXPORTED: enum.ALL_METHODS,
        },
    }

    def __init__(self, estimator: GradientBoostingClassifierClass):
        EstimatorBase.__init__(self, estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        try:
            est.estimators_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # Binary classifiers have one tree per stage and multiclass
        # classifiers one tree per class:
        trees, inits = get_stages_data(est)
        labels, label_kind = get_labels(est)
        self.meta_info = dict(
            n_estimators=len(trees),
            n_features=est.n_features_in_,  # ty: ignore[unresolved-attribute]
            n_outputs=1,  # the number of values of each leaf
            n_classes=len(est.classes_),
            n_trees_per_stage=len(inits),
            is_binary=len(inits) == 1,
            # The exponential loss (only binary) halves the raw predictions:
            is_exponential=est.loss == 'exponential',
            labels=labels,
            label_kind=label_kind,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(estimators=trees, inits=inits)
        L.info(f'Model data (keys): {self.model_data.keys()}')
        L.opt(lazy=True).debug(f'Model data: {self.model_data}')
