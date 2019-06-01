# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class EstimatorApiABC(ABC):
    """
    An abstract interface to ensure equal methods between the
    main class `sklearn_porter.Estimator` and all subclasses
    in `sklearn-porter.estimator.*`.
    """

    @abstractmethod
    def port(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            **kwargs
    ) -> str:
        pass
