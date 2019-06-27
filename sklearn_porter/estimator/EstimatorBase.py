# -*- coding: utf-8 -*-

from typing import Union, Set, Dict, Optional, Tuple
from pathlib import Path
from os import getcwd
from json import dumps

from sklearn.base import BaseEstimator

from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.utils import get_logger
from sklearn_porter.exceptions import NotSupportedYetError
from sklearn_porter.enums import Method, Language, Template

L = get_logger(__name__)


class EstimatorBase(EstimatorApiABC):

    DEFAULT_LANGUAGE = None  # type: Language
    DEFAULT_METHOD = None  # type: Method
    DEFAULT_TEMPLATE = None  # type: Template

    SUPPORT = None  # type: Dict[Language, Dict[Method, Set[Template]]]

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

        self.placeholders.update(dict(
            estimator_name=self.estimator_name,
            estimator_url=self.estimator_url,
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

        # Set default:
        method = method or self.DEFAULT_METHOD
        language = language or self.DEFAULT_LANGUAGE
        template = template or self.DEFAULT_TEMPLATE

        # Check method support:
        if method not in self.SUPPORT[language].keys():
            msg = 'Currently only `predict` ' \
                  'is a valid method type.'
            raise NotSupportedYetError(msg)

        # Check language support:
        if language not in self.SUPPORT.keys():
            msg = 'Currently the language `{}` ' \
                  'is not supported yet.'.format(language.value)
            raise NotSupportedYetError(msg)

        # Check the template support:
        if template not in self.SUPPORT[language][method]:
            msg = 'Currently the template `{}` ' \
                  'is not implemented yet.'.format(template.value)
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

        # Add extended base estimators:
        bases = list(set([base.__name__ for base in self.__class__.__bases__]))
        if 'EstimatorBase' in bases:
            bases.remove('EstimatorBase')
        if 'EstimatorApiABC' in bases:
            bases.remove('EstimatorApiABC')

        # Add desired estimator at the end:
        bases.append(self.__class__.__name__)

        for base_dir in bases:
            temps_dir = file_dir / base_dir / 'templates' / language.KEY
            if temps_dir.exists():
                temps_paths = set(temps_dir.glob('*.txt'))
                temps.update({path.stem: path.read_text()
                              for path in temps_paths})
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
