# -*- coding: utf-8 -*-

from typing import Union, Optional, Callable
from logging import Logger, ERROR

from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_logger


class DecisionTreeClassifier(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port a DecisionTreeClassifier classifier.

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
        super().__init__(estimator)
        self.logger = get_logger(__name__, logger=logger)
        self.logger.info('Create specific estimator `%s`.', self.estimator_name)
        est = self.estimator  # alias

        # TODO: Export and prepare model data from estimator.

    def port(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',  # language
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        temps = self.load_templates(language=to)

        print(temps.keys())

        return str(self.estimator)
