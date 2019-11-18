# -*- coding: utf-8 -*-

from os import environ, getcwd
from pathlib import Path
from time import sleep
from typing import Dict, Optional, Set, Tuple, Union, Callable

from jinja2 import DictLoader, Environment
from loguru import logger as L

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator

# sklearn-porter
from sklearn_porter import __version__ as sklearn_porter_version
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC


class EstimatorBase(EstimatorApiABC):

    SKLEARN_BASE_URL = 'https://scikit-learn.org/stable/modules/generated/'
    SKLEARN_URL = None  # type: str

    DEFAULT_LANGUAGE = None  # type: enum.Language
    DEFAULT_METHOD = None  # type: enum.Method
    DEFAULT_TEMPLATE = None  # type: enum.Template

    SUPPORT = None  # type: Dict[enum.Language, Dict[enum.Template, Set[enum.Method]]]

    estimator = None  # type: BaseEstimator
    estimator_name = None  # type: str
    estimator_url = None  # type: str

    model_data = {}  # stores parameters, e.g. weights or coefficients
    meta_info = {}  # stores meta information, e.g. num of classes or features

    placeholders = {}  # stores rendered templates and single information

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

        # Set name:
        self.estimator_name = estimator.__class__.__qualname__

        # Set URL:
        if self.SKLEARN_URL:
            url = self.SKLEARN_BASE_URL + self.SKLEARN_URL
        else:
            url = 'https://scikit-learn.org/stable/documentation.html'
        self.estimator_url = url

        # Add basic information:
        self.placeholders.update(
            dict(
                estimator_name=self.estimator_name,
                estimator_url=self.estimator_url,
                sklearn_version=sklearn_version,
                sklearn_porter_version=sklearn_porter_version,
            )
        )

        # Is it a test?
        self.placeholders.update(
            dict(is_test='SKLEARN_PORTER_PYTEST' in environ)
        )

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ):
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
        msg = (
            'You have to overwrite this method '
            '`port` in the class of the estimator.'
        )
        raise NotImplementedError(msg)

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

        if not directory:
            directory = Path(getcwd()).resolve()
        if isinstance(directory, str):
            directory = Path(directory)
        if not directory.is_dir():
            directory = directory.parent

        # Port/Transpile estimator:
        ported = self.port(
            language=language,
            template=template,
            class_name=class_name,
            converter=converter,
            to_json=to_json
        )
        if not isinstance(ported, tuple):
            ported = (ported, )

        # Dump ported estimator:
        suffix = language.value.SUFFIX
        filename = class_name + '.' + suffix
        filepath = directory / filename
        filepath.write_text(ported[0], encoding='utf-8')
        while not filepath.exists():
            sleep(0.01)
        paths = str(filepath)

        # Dump model data:
        if template == enum.Template.EXPORTED and len(ported) == 2:
            json_path = directory / (class_name + '.json')
            json_path.write_text(ported[1], encoding='utf-8')
            while not json_path.exists():
                sleep(0.01)
            paths = (paths, str(json_path))

        return paths

    def _load_templates(
        self, language: Union[str, enum.Language]
    ) -> Environment:
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
        language = (
            enum.Language[language.upper()].value
            if isinstance(language, str) else language
        )

        tpls = {}  # Dict

        # 1. Load basic language templates (e.g. `if`, `else`, ...):
        lang_tpls = language.TEMPLATES
        if isinstance(language.TEMPLATES, dict):
            tpls.update(lang_tpls)
        L.debug(
            'Load template variables: {}'.format(', '.join(lang_tpls.keys()))
        )

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
                tpls.update(
                    {path.stem: path.read_text()
                     for path in tpls_paths}
                )

        L.debug('Load template files: {}'.format(', '.join(tpls.keys())))

        environment = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            loader=DictLoader(tpls),
        )
        return environment
