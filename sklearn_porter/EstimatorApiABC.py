# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Optional, List

from pathlib import Path


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

    @abstractmethod
    def dump(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            directory: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Union[str, List[str]]:
        pass
