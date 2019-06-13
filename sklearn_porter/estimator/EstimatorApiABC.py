# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple

from pathlib import Path

from sklearn_porter.enums import Method, Language, Template


class EstimatorApiABC(ABC):
    """
    An abstract interface to ensure equal methods between the
    main class `sklearn_porter.Estimator` and all subclasses
    in `sklearn-porter.estimator.*`.
    """

    @abstractmethod
    def port(
            self,
            method: Method,
            language: Language,
            template: Template,
            **kwargs
    ) -> Union[str, Tuple[str, str]]:
        pass

    @abstractmethod
    def dump(
            self,
            method: Method,
            language: Language,
            template: Template,
            directory: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Union[str, Tuple[str, str]]:
        pass
