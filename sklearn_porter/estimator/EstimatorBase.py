import sys
from collections.abc import Callable
from os import environ
from pathlib import Path
from time import sleep
from typing import Any, cast

from jinja2 import DictLoader, Environment, Template
from loguru import logger as L

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator

from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC

# sklearn-porter
from sklearn_porter.meta import __version__ as sklearn_porter_version
from sklearn_porter.utils import tabify


class _Template(Template):
    """
    Template which renders only single blocks of the main template (e.g.
    `attached.class`), if the global `_blocks` of the environment names them.
    A pipeline uses it to insert the classes and the creation of its
    transformers into the ported final estimator.
    """

    def render(self, *args, **kwargs):
        blocks = cast(tuple, self.environment.globals.get('_blocks'))
        if not blocks or not str(self.name).endswith('.class'):
            return super().render(*args, **kwargs)
        context = self.new_context(dict(*args, **kwargs))
        return {
            name: ''.join(self.blocks[name](context))
            for name in blocks
            if name in self.blocks
        }


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
    # Additional globals of the templates (e.g. of a pipeline, see `Pipeline`):
    template_globals: dict[str, Any]

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.model_data = {}
        self.meta_info = {}
        self.placeholders = {}
        self.template_globals = {}

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
        if template == enum.Template.COMBINED:
            ported = (tabify(ported[0]),) + ported[1:]

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

        Template files: `sklearn_porter/estimator/SVC/templates/<language>/*`
        (or `sklearn_porter/transformer/<Name>/templates/<language>/*`)
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
        # (also of the bases of its bases). The templates are in the
        # directory of the module of each class (e.g. of an estimator in
        # `sklearn_porter/estimator/` or of a transformer in
        # `sklearn_porter/transformer/`):
        ignored = {'EstimatorBase', 'EstimatorApiABC', 'TransformerBase'}
        bases = [
            b
            for b in reversed(self.__class__.__mro__)
            if b.__module__.startswith('sklearn_porter.')
            and b.__name__ not in ignored
        ]
        for base in bases:
            module_path = sys.modules[base.__module__].__file__
            if not module_path:
                continue
            tpls_dir = Path(module_path).parent / 'templates' / lang.KEY
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
        environment.template_class = _Template
        # The labels are available in all templates, also in the included
        # template `labels` of the language:
        environment.globals.update(self._get_label_placeholders(language))
        environment.globals.update(self.template_globals)
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
