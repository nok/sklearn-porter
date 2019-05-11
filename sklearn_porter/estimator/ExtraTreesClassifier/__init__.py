# -*- coding: utf-8 -*-

from typing import Union, Optional, Callable
from logging import Logger, ERROR

from sklearn.ensemble.forest import ExtraTreesClassifier \
    as ExtraTreesClassifierClass

from sklearn_porter.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.utils import get_logger


class ExtraTreesClassifier(EstimatorBase, EstimatorApiABC):
    """
    Extract model data and port an ExtraTreesClassifier classifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """
    estimator = None  # type: ExtraTreesClassifierClass

    def __init__(
            self,
            estimator: ExtraTreesClassifierClass,
            logger: Union[Logger, int] = ERROR
    ):
        super().__init__(self.__class__.__qualname__)
        self.logger = get_logger(__name__, logger=logger)
        self.logger.info('Create specific estimator `%s`.', self.estimator_name)

        self.estimator = est = estimator  # alias

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
