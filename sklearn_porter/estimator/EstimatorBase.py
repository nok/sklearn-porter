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

    DEFAULT_LANGUAGE = None  # type: Language
    DEFAULT_METHOD = None  # type: Method
    DEFAULT_TEMPLATE = None  # type: Template

    SUPPORT = None  # type: Dict[Language, Dict[Method, Set[Template]]]

    estimator = None  # type: BaseEstimator

    model_data = {}  # stores parameters, e.g. weights or coefficients
    meta_info = {}  # stores meta information, e.g. num of classes or features

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__qualname__

    def port(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None,
            **kwargs
    ):
        msg = 'You have to overwrite this method ' \
              '`port` in the class of the estimator.'
        raise NotImplementedError(msg)

    def check(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None
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

        Returns
        -------
        The ported estimator.
        """

        # Check estimator defaults:
        if not self.SUPPORT:
            msg = 'You have to update the support ' \
                  'matrix in the class of the estimator.'
            raise NotImplementedError(msg)
        if not self.DEFAULT_METHOD:
            msg = 'You have to set a default method ' \
                  'in the class of the estimator.'
            raise NotImplementedError(msg)
        if not self.DEFAULT_LANGUAGE:
            msg = 'You have to set a default language ' \
                  'in the class of the estimator.'
            raise NotImplementedError(msg)
        if not self.DEFAULT_TEMPLATE:
            msg = 'You have to set a default template ' \
                  'in the class of the estimator.'
            raise NotImplementedError(msg)

        # Set default:
        method = method or self.DEFAULT_METHOD
        language = language or self.DEFAULT_LANGUAGE
        template = template or self.DEFAULT_TEMPLATE

        # Check method support:
        if method not in self.SUPPORT[language].keys():
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotSupportedError(msg)

        # Check language support:
        if language not in self.SUPPORT.keys():
            msg = 'Currently the language `{}` ' \
                  'is not supported yet.'.format(language.value)
            raise NotSupportedError(msg)

        # Check the template support:
        if template not in self.SUPPORT[language][method]:
            msg = 'Currently the template `{}` ' \
                  'is not implemented yet.'.format(template.value)
            raise NotSupportedError(msg)

        return method, language, template

    def dump(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None,
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

        method, language, template = self.check(
            method=method, language=language, template=template)

        class_name = kwargs.get('class_name')

        # Port/Transpile estimator:
        ported = self.port(
            method=method,
            language=language,
            template=template,
            **kwargs
        )
        if not isinstance(ported, tuple):
            ported = (ported, )

        # Dump ported estimator:
        suffix = language.value.SUFFIX
        filename = class_name + '.' + suffix
        filepath = directory / filename
        filepath.write_text(ported[0], encoding='utf-8')
        paths = str(filepath)

        # Dump model data:
        if template == Template.EXPORTED and len(ported) == 2:
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
