# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

# sklearn-porter
from sklearn_porter import enums as enum


class EstimatorApiABC(ABC):
    """
    An abstract interface to ensure equal methods between the
    main class `sklearn_porter.Estimator` and all subclasses
    in `sklearn-porter.estimator.*`.
    """
    @abstractmethod
    def port(
        self,
        language: Optional[enum.Language] = None,
        template: Optional[enum.Template] = None,
        to_json: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        to_json : bool (default: False)
            Return the result as JSON string.
        kwargs

        Returns
        -------
        The ported estimator.
        """
    @abstractmethod
    def save(
        self,
        language: Optional[enum.Language] = None,
        template: Optional[enum.Template] = None,
        directory: Optional[Union[str, Path]] = None,
        to_json: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Dump an estimator to the filesystem.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template
        directory : str or Path
            The destination directory.
        to_json : bool (default: False)
            Return the result as JSON string.
        kwargs

        Returns
        -------
        The paths to the dumped files.
        """
