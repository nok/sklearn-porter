# -*- coding: utf-8 -*-

from typing import Union, Set, Dict
from pathlib import Path
from json import dumps

from sklearn.base import BaseEstimator

from sklearn_porter.utils import get_logger

L = get_logger(__name__)


class EstimatorBase:

    estimator = None  # type: BaseEstimator

    supported_languages = None  # type: Set
    supported_methods = None  # type: Set
    supported_templates = None  # type: Set

    model_data = {}  # stores parameters, e.g. weights or coefficients
    meta_info = {}  # stores meta information, e.g. num of classes or features

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__qualname__

    def check_arguments(
            self,
            method: str,
            language: str,
            template: str
    ):
        self._check_method(method)
        self._check_language(language)
        self._check_template(template)

    def _check_method(self, method: str):
        """
        Check whether `method` is in `self.supported_methods` or not.

        Parameters
        ----------
        method : str
            The kind of method.

        Returns
        -------
            If the check fails an exception will be raised.
        """
        if not self.supported_methods or \
                not method in self.supported_methods:
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotImplementedError(msg)

    def _check_language(self, language: str):
        """
        Check whether `language` is in `self.supported_languages` or not.

        Parameters
        ----------
        language : str
            The kind of method.

        Returns
        -------
            If the check fails an exception will be raised.
        """
        if not self.supported_languages or \
                not language in self.supported_languages:
            msg = 'Currently the language `{}` ' \
                  'is not implemented yet.'.format(language)
            raise NotImplementedError(msg)

    def _check_template(self, template: str):
        """
        Check whether `template` is in `self.supported_templates` or not.

        Parameters
        ----------
        template : str
            The kind of method.

        Returns
        -------
            If the check fails an exception will be raised.
        """
        if not self.supported_templates or \
                not template in self.supported_templates:
            msg = 'currently the template `{}` ' \
                  'is not implemented yet.'.format(template)
            raise NotImplementedError(msg)

    def _load_templates(self, language: str) -> Dict:
        """
        Load templates from static files and the global language files.

        Template files: `sklearn_porter/estimator/SVC/templates/<language>/*txt`
        Language files: `sklearn_porter/language/<language>.py`

        Parameters
        ----------
        language : str
            The passed programming language.

        Returns
        -------
        temps : dict
            A dictionary with all loaded templates.
        """
        temps = {}

        # 1. Load from template files:
        file_dir = Path(__file__).parent
        temps_dir = file_dir / self.estimator_name / 'templates' / language
        if temps_dir.exists():
            temps_paths = set(temps_dir.glob('*.txt'))
            temps.update({path.stem: path.read_text() for path in temps_paths})
        L.debug('Load template files: {}'.format(', '.join(temps.keys())))

        # 2. Load from language files:
        #    The next three lines are similar to this import statement:
        #    `from sklearn_porter.language.java import TEMPLATES as lang_temps`
        package = 'sklearn_porter.language.' + language
        name = 'TEMPLATES'
        lang_temps = getattr(__import__(package, fromlist=[name]), name)

        if isinstance(lang_temps, dict):
            temps.update(lang_temps)
        L.debug('Load template variables: {}'.format(', '.join(lang_temps.keys())))

        return temps

    @staticmethod
    def _dump_dict(
            obj: Dict,
            sort_keys: bool = True,
            indent: Union[bool, int] = 2
    ) -> str:
        indent = indent if indent else None
        return dumps(obj, sort_keys=sort_keys, indent=indent)
