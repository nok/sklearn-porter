# -*- coding: utf-8 -*-

from typing import Union, Optional, Callable
from logging import Logger, ERROR

from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.EstimatorInterApiABC import EstimatorInterApiABC
from sklearn_porter.utils import get_logger


class DecisionTreeClassifier(EstimatorInterApiABC):
    """
    Port a DecisionTreeClassifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    estimator = None  # type: DecisionTreeClassifierClass

    def __init__(
            self,
            estimator: DecisionTreeClassifierClass,
            logger: Union[Logger, int] = ERROR
    ):
        self.logger = get_logger(__name__, logger=logger)

        self.estimator = est = estimator
        self.default_class_name = estimator.__class__.__name__
        self.logger.info('Start extracting model data from `%s`.',
                         self.default_class_name)

    def port(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        return str(self.estimator)
