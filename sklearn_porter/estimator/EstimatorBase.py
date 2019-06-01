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

    def _load_templates(self, language: str) -> Dict:
        temps = {}

        # 1. Load special templates from files:
        file_dir = Path(__file__).parent
        temps_dir = file_dir / self.estimator_name / 'templates' / language
        if temps_dir.exists():
            temps_paths = set(temps_dir.glob('*.txt'))
            temps.update({path.stem: path.read_text() for path in temps_paths})
        L.debug('Load template files: {}'.format(', '.join(temps.keys())))

        # 2. Load basic templates from package:
        # from sklearn_porter.language.java import TEMPLATES as lang_temps
        package = 'sklearn_porter.language.' + language
        name = 'TEMPLATES'
        lang_temps = getattr(__import__(package, fromlist=[name]), name)
        if isinstance(lang_temps, dict):
            temps.update(lang_temps)
        L.debug('Load template variables: {}'.format(', '.join(lang_temps.keys())))

        return temps

    def _check_method(self, method: str):
        if not self.supported_methods or \
                not method in self.supported_methods:
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotImplementedError(msg)

    def _check_language(self, language: str):
        if not self.supported_languages or \
                not language in self.supported_languages:
            msg = 'Currently the language `{}` ' \
                  'is not implemented yet.'.format(language)
            raise NotImplementedError(msg)

    def _check_template(self, template: str):
        if not self.supported_templates or \
                not template in self.supported_templates:
            msg = 'currently the template `{}` ' \
                  'is not implemented yet.'.format(template)
            raise NotImplementedError(msg)

    @staticmethod
    def _dump_dict(
            obj: Dict,
            sort_keys: bool = True,
            indent: Union[bool, int] = 2
    ) -> str:
        indent = indent if indent else None
        return dumps(obj, sort_keys=sort_keys, indent=indent)
