# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

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
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `DecisionTreeClassifier`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
    @abstractmethod
    def save(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        directory: Optional[Union[str, Path]] = None,
        to_json: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        Dump an estimator to the filesystem.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `DecisionTreeClassifier`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        directory : str or Path
            The destination directory.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The paths to the dumped files.
        """
