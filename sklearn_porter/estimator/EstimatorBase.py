# -*- coding: utf-8 -*-

from typing import Union, Set, Dict
from pathlib import Path
from json import dumps

from sklearn.base import BaseEstimator

from sklearn_porter.utils import get_logger
from sklearn_porter.exceptions import NotSupportedError
from sklearn_porter.enums import Method, Language, Template

L = get_logger(__name__)


class EstimatorBase:

    estimator = None  # type: BaseEstimator

    model_data = {}  # stores parameters, e.g. weights or coefficients
    meta_info = {}  # stores meta information, e.g. num of classes or features

    support = None  # type: Dict[Language, Dict[Method, Set[Template]]]

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__qualname__

    def check_arguments(
            self,
            method: Method,
            language: Language,
            template: Template
    ):
        """
        Check whether the passed arguments are supported by the current
        implementation of estimator or not. For that each estimator
        has to overwrite the internal variables `self.supported_methods`,
        `self.supported_languages` and `self.supported_templates` by
        setting the current implementations.

        Parameters
        ----------
        method : str
            The passed method name.
        language : str
            The passed language name.
        template : str
            The passed template name.

        Returns
        -------
        If a check fails an exception will be raised.
        """

        # Language
        if language not in self.support.keys():
            msg = 'Currently the language `{}` ' \
                  'is not supported yet.'.format(language)
            raise NotSupportedError(msg)

        # Method:
        if method not in self.support[language].keys():
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotSupportedError(msg)

        # Template:
        if template not in self.support[language][method]:
            msg = 'currently the template `{}` ' \
                  'is not implemented yet.'.format(template)
            raise NotSupportedError(msg)

    def _load_templates(self, language: Union[str, Language]) -> Dict:
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
        language = Language[language.upper()] if \
            isinstance(language, str) else language
        language = language.value

        temps = {}

        # 1. Load from template files:
        file_dir = Path(__file__).parent
        temps_dir = file_dir / self.estimator_name / 'templates' / language.KEY
        if temps_dir.exists():
            temps_paths = set(temps_dir.glob('*.txt'))
            temps.update({path.stem: path.read_text() for path in temps_paths})
        L.debug('Load template files: {}'.format(', '.join(temps.keys())))

        # 2. Load from language files:
        lang_temps = language.TEMPLATES
        if isinstance(language.TEMPLATES, dict):
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
