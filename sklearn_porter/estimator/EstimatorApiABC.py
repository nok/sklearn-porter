# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

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
        """
        Port an estimator.

        Parameters
        ----------
        method : Method
            The required method.
        language : Language
            The required language.
        template : Template
            The required template.
        kwargs

        Returns
        -------
        The ported estimator.
        """
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
        """
        Dump an estimator to the filesystem.

        Parameters
        ----------
        method : Method
            The required method.
        language : Language
            The required language.
        template : Template
            The required template
        directory : str or Path
            The destination directory.
        kwargs

        Returns
        -------
        The paths to the dumped files.
        """
        pass
