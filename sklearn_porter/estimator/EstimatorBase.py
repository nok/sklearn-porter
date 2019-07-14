# -*- coding: utf-8 -*-

from typing import Union, Set, Dict, Optional, Tuple
from pathlib import Path
from os import getcwd, environ
from json import dumps

from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from jinja2 import Environment, DictLoader

from sklearn_porter import __version__ as sklearn_porter_version
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.utils import get_logger
from sklearn_porter.exceptions import NotSupportedYetError
from sklearn_porter.enums import Method, Language, Template

L = get_logger(__name__)


class EstimatorBase(EstimatorApiABC):

    DEFAULT_LANGUAGE = None  # type: Language
    DEFAULT_METHOD = None  # type: Method
    DEFAULT_TEMPLATE = None  # type: Template

    SUPPORT = None  # type: Dict[Language, Dict[Template, Set[Method]]]

    estimator = None  # type: BaseEstimator
    estimator_name = None  # type: str
    estimator_url = None  # type: str

    model_data = {}  # stores parameters, e.g. weights or coefficients
    meta_info = {}  # stores meta information, e.g. num of classes or features

    placeholders = {}  # stores rendered templates and single information

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__qualname__

        default_url = 'https://scikit-learn.org/stable/documentation.html'
        self.estimator_url = default_url
        urls = {
            'AdaBoostClassifier': 'ensemble.AdaBoostClassifier',
            'BernoulliNB': 'naive_bayes.BernoulliNB',
            'DecisionTreeClassifier': 'tree.DecisionTreeClassifier',
            'ExtraTreesClassifier': 'ensemble.ExtraTreesClassifier',
            'GaussianNB': 'naive_bayes.GaussianNB',
            'KNeighborsClassifier': 'neighbors.KNeighborsClassifier',
            'LinearSVC': 'svm.LinearSVC',
            'MLPClassifier': 'neural_network.MLPClassifier',
            'MLPRegressor': 'neural_network.MLPRegressor',
            'NuSVC': 'svm.NuSVC',
            'RandomForestClassifier': 'ensemble.RandomForestClassifier',
            'SVC': 'svm.SVC',
        }
        if self.estimator_name in urls:
            base_url = 'https://scikit-learn.org/stable/' \
                       'modules/generated/sklearn.{}.html'
            self.estimator_url = base_url.format(urls.get(self.estimator_name))

        # Add base information:
        self.placeholders.update(dict(
            estimator_name=self.estimator_name,
            estimator_url=self.estimator_url,
            sklearn_version=sklearn_version,
            sklearn_porter_version=sklearn_porter_version,
        ))

        # Is it a test?
        self.placeholders.update(dict(
            is_test='PYTEST_CURRENT_TEST' in environ,
        ))

    def port(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None,
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

        Returns
        -------
        The ported estimator.
        """
        msg = 'You have to overwrite this method ' \
              '`port` in the class of the estimator.'
        raise NotImplementedError(msg)

    def check(
            self,
            method: Optional[Method] = None,
            language: Optional[Language] = None,
            template: Optional[Template] = None
    ) -> Tuple[Method, Language, Template]:
        """
        Check whether the passed values (kind of method, language
        and template) are supported by the estimator or not.
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

        # Check language support:
        language = language or self.DEFAULT_LANGUAGE
        if language not in self.SUPPORT.keys():
            msg = 'Currently the language `{}` ' \
                  'is not supported yet.'.format(language.value)
            raise NotSupportedYetError(msg)

        # Check the template support:
        template = template or self.DEFAULT_TEMPLATE
        if template not in self.SUPPORT[language].keys():
            msg = 'Currently the template `{}` ' \
                  'is not implemented yet.'.format(template.value)
            raise NotSupportedYetError(msg)

        # Check method support:
        method = method or self.DEFAULT_METHOD
        if method not in self.SUPPORT[language][template]:
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotSupportedYetError(msg)

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

    def _load_templates(self, language: Union[str, Language]) -> Environment:
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
        environment : Environment
            An Jinja environment with all loaded templates.
        """
        language = Language[language.upper()].value if \
            isinstance(language, str) else language

        tpls = {}  # Dict

        # 1. Load basic language templates (e.g. `if`, `else`, ...):
        lang_tpls = language.TEMPLATES
        if isinstance(language.TEMPLATES, dict):
            tpls.update(lang_tpls)
        L.debug('Load template variables: {}'.format(
            ', '.join(lang_tpls.keys())))

        # 2. Load base language templates (e.g. `base.attached.class`):
        root_dir = Path(__file__).parent.parent
        tpls_dir = root_dir / 'language' / language.LABEL / 'templates'
        if tpls_dir.exists():
            tpls_paths = set(tpls_dir.glob('*.jinja2'))
            tpls.update({path.stem: path.read_text() for path in tpls_paths})

        # 3. Load specific templates from template files:
        bases = list(set([base.__name__ for base in self.__class__.__bases__]))
        if 'EstimatorBase' in bases:
            bases.remove('EstimatorBase')
        if 'EstimatorApiABC' in bases:
            bases.remove('EstimatorApiABC')
        bases.append(self.__class__.__name__)
        est_dir = root_dir / 'estimator'
        for base_dir in bases:
            tpls_dir = est_dir / base_dir / 'templates' / language.KEY
            if tpls_dir.exists():
                tpls_paths = set(tpls_dir.glob('*.jinja2'))
                tpls.update({path.stem: path.read_text()
                             for path in tpls_paths})

        L.debug('Load template files: {}'.format(', '.join(tpls.keys())))

        environment = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            loader=DictLoader(tpls),
        )
        return environment

    @staticmethod
    def _dump_dict(
            obj: Dict,
            sort_keys: bool = True,
            indent: Union[bool, int] = 2
    ) -> str:
        indent = indent if indent else None
        return dumps(obj, sort_keys=sort_keys, indent=indent)
