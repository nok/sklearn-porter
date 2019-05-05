# -*- coding: utf-8 -*-

from typing import Union, Optional, Callable
from abc import ABC, abstractmethod


class EstimatorInterApiABC(ABC):
    """
    An abstract interface to ensure equal methods between the
    main class `sklearn_porter.Estimator` and all subclasses
    like `sklearn-porter.estimator.DecisionTreeClassifier`.
    """

    @abstractmethod
    def port(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        pass
