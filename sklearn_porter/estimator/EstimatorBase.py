# -*- coding: utf-8 -*-

from typing import Union, Set, Dict, Optional, Tuple
from pathlib import Path
from os import getcwd
from json import dumps

from sklearn.base import BaseEstimator

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.utils import get_logger
from sklearn_porter.exceptions import NotSupportedError
from sklearn_porter.enums import Method, Language, Template

L = get_logger(__name__)


class EstimatorBase(EstimatorApiABC):

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
        implementation of the estimator or not. For that each estimator
        has to overwrite the internal variable `self.support`.

        Parameters
        ----------
        method : Method
            The required method.
        language : Language
            The required language.
        template : Template
            The required template.

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

    def port(
            self,
            method: Method,
            language: Language,
            template: Template,
            **kwargs
    ):
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
        msg = 'You have to overwrite method `port` in the estimator class'
        raise NotImplementedError(msg)

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

        if not directory:
            directory = Path(getcwd()).resolve()
        if isinstance(directory, str):
            directory = Path(directory)
        if not directory.is_dir():
            directory = directory.parent

        class_name = kwargs.get('class_name')

        # Port/Transpile estimator:
        ported = self.port(method, language, template, **kwargs)
        if not isinstance(ported, tuple):
            ported = (ported, )

        # Dump ported estimator:
        suffix = language.value.SUFFIX
        filename = class_name + '.' + suffix
        filepath = directory / filename
        filepath.write_text(ported[0], encoding='utf-8')
        paths = str(filepath)

        # Dump model data:
        if template == 'exported' and len(ported) == 2:
            json_path = directory / (class_name + '.json')
            json_path.write_text(ported[1], encoding='utf-8')
            paths = (paths, str(json_path))

        return paths

    def _load_templates(self, language: Union[str, Language]) -> Dict:
        """
        Load templates from static files and the global language files.

        Template files: `sklearn_porter/estimator/SVC/templates/<language>/*txt`
        Language files: `sklearn_porter/language/<language>.py`

        Parameters
        ----------
        language : str or Language
            The required language.

        Returns
        -------
        temps : dict
            A dictionary with all loaded templates.
        """
        language = Language[language.upper()].value if \
            isinstance(language, str) else language

        temps = {}

        # 1. Load default templates from language files:
        lang_temps = language.TEMPLATES
        if isinstance(language.TEMPLATES, dict):
            temps.update(lang_temps)
        L.debug('Load template variables: {}'.format(
            ', '.join(lang_temps.keys())))

        # 2. Load specific templates from template files:
        file_dir = Path(__file__).parent
        temps_dir = file_dir / self.estimator_name / 'templates' / language.KEY
        if temps_dir.exists():
            temps_paths = set(temps_dir.glob('*.txt'))
            temps.update({path.stem: path.read_text() for path in temps_paths})
        L.debug('Load template files: {}'.format(', '.join(temps.keys())))

        return temps

    @staticmethod
    def _dump_dict(
            obj: Dict,
            sort_keys: bool = True,
            indent: Union[bool, int] = 2
    ) -> str:
        indent = indent if indent else None
        return dumps(obj, sort_keys=sort_keys, indent=indent)
