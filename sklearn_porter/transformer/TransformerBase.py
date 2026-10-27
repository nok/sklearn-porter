from sklearn.base import BaseEstimator

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class TransformerBase(EstimatorBase):
    """
    Base class of all ported transformers.

    A ported transformer has the method `transform`, which returns the
    transformed features of one sample. Its `main` prints the transformed
    features of each sample in one line (separated by spaces, with the shared
    template `values` of each language), so the output can be passed to a
    ported estimator and `make()` reads it without JSON.
    """

    DEFAULT_METHOD = enum.Method.TRANSFORM

    def __init__(self, transformer: BaseEstimator):
        super().__init__(transformer)
        # The wording of the base templates:
        self.placeholders.update(
            dict(
                kind='Transformer',
                action='a transformation',
                actions='the transformations',
            )
        )
