from collections.abc import Callable
from os import environ
from pathlib import Path
from time import sleep
from typing import Any

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
    SKLEARN_URL: str | None = None

    DEFAULT_LANGUAGE: enum.Language
    DEFAULT_METHOD: enum.Method
    DEFAULT_TEMPLATE: enum.Template

    SUPPORT: dict[enum.Language, dict[enum.Template, set[enum.Method]]]

    estimator: BaseEstimator
    estimator_name: str
    estimator_url: str

    model_data: dict[str, Any]  # parameters, e.g. weights or coefficients
    meta_info: dict[str, Any]  # meta information, e.g. num of features
    placeholders: dict[str, Any]  # rendered templates and single information

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.model_data = {}
        self.meta_info = {}
        self.placeholders = {}

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
        directory: str | Path | None = None,
        to_json: bool = False,
    ) -> str | tuple[str, str]:
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
            directory = Path.cwd()
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
            to_json=to_json,
        )
        if not isinstance(ported, tuple):
            ported = (ported,)

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

    def _load_templates(self, language: str | enum.Language) -> Environment:
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
        if isinstance(language, str):
            language = enum.Language[language.upper()]
        lang = language.value

        tpls: dict[str, str] = {}

        # 1. Load basic language templates (e.g. `if`, `else`, ...):
        tpls.update(lang.TEMPLATES)
        L.debug(f'Load template variables: {", ".join(lang.TEMPLATES)}')

        # 2. Load base language templates (e.g. `base.attached.class`):
        root_dir = Path(__file__).parent.parent
        tpls_dir = root_dir / 'language' / lang.KEY / 'templates'
        if tpls_dir.exists():
            tpls_paths = set(tpls_dir.glob('*.jinja2'))
            tpls.update({path.stem: path.read_text() for path in tpls_paths})

        # 3. Load specific templates from template files:
        # The templates of a class overwrite the templates of its bases
        # (also of the bases of its bases):
        ignored = {'EstimatorBase', 'EstimatorApiABC'}
        bases = [b.__name__ for b in reversed(self.__class__.__mro__)]
        bases = [b for b in dict.fromkeys(bases) if b not in ignored]
        est_dir = root_dir / 'estimator'
        for base_dir in bases:
            tpls_dir = est_dir / base_dir / 'templates' / lang.KEY
            if tpls_dir.exists():
                tpls_paths = set(tpls_dir.glob('*.jinja2'))
                tpls.update(
                    {path.stem: path.read_text() for path in tpls_paths}
                )

        L.debug(f'Load template files: {", ".join(tpls)}')

        environment = Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            loader=DictLoader(tpls),
        )
        # The labels are available in all templates, also in the included
        # template `labels` of the language:
        environment.globals.update(self._get_label_placeholders(language))
        return environment

    def _get_label_placeholders(self, language: enum.Language) -> dict:
        """
        Render the labels of the classes and their type for a language.

        Parameters
        ----------
        language : Language
            The required language.

        Returns
        -------
        The placeholders `label_type` and `label_values`, or nothing if the
        estimator has no labels (e.g. a regressor).
        """
        kind = self.meta_info.get('label_kind')
        if not kind:
            return {}
        lang = language.value
        to_literal = lang.str_literal if kind == 'str' else repr
        values = map(to_literal, self.meta_info['labels'])
        return dict(
            label_type=lang.TEMPLATES[kind],
            label_values=', '.join(values),
        )
