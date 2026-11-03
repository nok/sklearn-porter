from collections.abc import Callable
from pathlib import Path
from typing import Any

# Additional installed modules:
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

# scikit-learn
from sklearn.base import BaseEstimator
from tabulate import tabulate

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter.Estimator import Estimator
from sklearn_porter.Pipeline import Pipeline
from sklearn_porter.Transformer import Transformer, is_transformer


class Porter:
    """
    Central class which ports a fitted estimator (a classifier, a regressor
    or an outlier detector), a fitted transformer (e.g. a `StandardScaler`)
    or a fitted pipeline of transformers and a final estimator of
    scikit-learn.

    It creates a `Transformer` for a transformer, a `Pipeline` for a
    pipeline whose steps are all supported and otherwise an `Estimator`,
    and forwards all calls to it.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        class_name: str | None = None,
        converter: Callable[[object], str] | None = None,
        ignore_preprocessing: bool = False,
    ):
        """
        Validate and coordinate the passed estimator, transformer or pipeline.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted estimator, transformer or pipeline of scikit-learn.
        language : str or Language, optional
            The target programming language. By default the default language
            of the estimator will be used.
        template : str or Template, optional
            The template of the generated code. By default the default
            template of the estimator will be used.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `DecisionTreeClassifier`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        ignore_preprocessing : bool, default=False
            Port only the final estimator of a pipeline with preprocessing
            steps (see `Estimator`), even if the steps could be ported.

        Raises
        ------
        NotSupportedYetError
            If the passed pipeline contains preprocessing steps which can't
            be ported and `ignore_preprocessing` is False.
        """
        self._ported: Estimator | Transformer | Pipeline
        if is_transformer(estimator):
            self._ported = Transformer(
                estimator,
                language=language,
                template=template,
                class_name=class_name,
                converter=converter,
            )
        elif not ignore_preprocessing and Pipeline.supports(estimator):
            self._ported = Pipeline(
                estimator,
                language=language,
                template=template,
                class_name=class_name,
                converter=converter,
            )
        else:
            # Also a pipeline (or a search) with unsupported (or ignored)
            # preprocessing steps, see `ignore_preprocessing`:
            self._ported = Estimator(
                estimator,
                language=language,
                template=template,
                class_name=class_name,
                converter=converter,
                ignore_preprocessing=ignore_preprocessing,
            )

    @property
    def ported(self) -> Estimator | Transformer | Pipeline:
        """The created `Estimator`, `Transformer` or `Pipeline`."""
        return self._ported

    @property
    def estimator(self):
        return self._ported.estimator

    @property
    def language(self) -> str:
        return self._ported.language

    @language.setter
    def language(self, language: str | enum.Language):
        self._ported.language = language

    @property
    def template(self) -> str:
        return self._ported.template

    @template.setter
    def template(self, template: str | enum.Template):
        self._ported.template = template

    @property
    def class_name(self) -> str:
        return self._ported.class_name

    @class_name.setter
    def class_name(self, class_name: str):
        self._ported.class_name = class_name

    @property
    def converter(self) -> Callable[[object], str]:
        return self._ported.converter

    @converter.setter
    def converter(self, converter: Callable[[object], str]):
        self._ported.converter = converter

    def can(
        self,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        method: str | enum.Method | None = None,
    ) -> bool:
        return self._ported.can(
            language=language, template=template, method=method
        )

    def port(self, to_json: bool = False, **kwargs) -> str | tuple[str, str]:
        """
        Port or transpile the passed estimator to a target programming
        language (see `Estimator.port`, `Transformer.port` and
        `Pipeline.port`).
        """
        return self._ported.port(to_json=to_json, **kwargs)

    def save(
        self,
        directory: str | Path | None = None,
        to_json: bool = False,
        **kwargs,
    ) -> str | tuple[str, str]:
        """
        Port the passed estimator to a target programming language and save
        it (see `Estimator.save`, `Transformer.save` and `Pipeline.save`).
        """
        return self._ported.save(directory=directory, to_json=to_json, **kwargs)

    def make(
        self,
        x: ArrayLike | spmatrix,
        **kwargs,
    ) -> tuple[Any, Any] | np.ndarray:
        """
        Make predictions (`Estimator.make` and `Pipeline.make`) or transform
        the samples (`Transformer.make`) with the transpiled estimator
        locally.
        """
        return self._ported.make(x, **kwargs)

    def predict(
        self,
        x: ArrayLike | spmatrix,
        **kwargs,
    ) -> tuple[Any, Any]:
        """
        Make predictions with the transpiled estimator or pipeline locally
        (see `Estimator.make` and `Pipeline.make`).

        Raises
        ------
        AttributeError
            If the passed estimator is a transformer.
        """
        if isinstance(self._ported, Transformer):
            msg = (
                f'The transformer `{self._ported._name}` '
                "can't predict, use `transform` instead."
            )
            raise AttributeError(msg)
        return self._ported.make(x, **kwargs)

    def transform(self, x: ArrayLike | spmatrix, **kwargs) -> np.ndarray:
        """
        Transform the samples with the transpiled transformer locally (see
        `Transformer.make`).

        Raises
        ------
        AttributeError
            If the passed estimator isn't a transformer.
        """
        if not isinstance(self._ported, Transformer):
            msg = (
                f'The {self._ported.KIND} `{self._ported._name}` '
                "can't transform, use `predict` instead."
            )
            raise AttributeError(msg)
        return self._ported.make(x, **kwargs)

    def test(self, x, **kwargs) -> float:
        """
        Compute the integrity score of the ported estimator (see
        `Estimator.test`, `Transformer.test` and `Pipeline.test`).
        """
        return self._ported.test(x, **kwargs)

    def __repr__(self):
        return repr(self._ported)


def show(language: str | enum.Language | None = None) -> str:
    """
    Show the supported estimators and transformers, programming languages
    and templates in two compact tables.

    Parameters
    ----------
    language : Language
        The requested programming language.

    Returns
    -------
    The tables of the estimators and transformers and a legend.
    """
    languages = enum.LANGUAGES
    if language:
        language = enum.Language.convert(language)
        languages = {language.value.KEY: language.value}
    templates = dict(attached='ᴀ', combined='ᴄ', exported='ᴇ')
    tables = []
    for title, clazzes in (
        ('Estimator', Estimator.estimators()),
        ('Transformer', Transformer.transformers()),
    ):
        # The classes (not instances, e.g. a `ColumnTransformer` needs
        # arguments) by their public names:
        clazzes = dict(sorted((_get_public_name(c), c) for c in clazzes))
        table = []
        for name, est in clazzes.items():
            tr = [name]
            for lang in languages.keys():
                td = []
                for tpl in templates.keys():
                    if can(
                        estimator=est,
                        language=lang,
                        template=tpl,
                        method='predict_proba',
                    ):
                        out = f'✓{templates.get(tpl)}ᴾ'
                    elif can(estimator=est, language=lang, template=tpl):
                        out = f'✓{templates.get(tpl)} '
                    else:
                        out = '···'
                    td.append(out)
                tr.append(' | '.join(td))
            table.append(tr)
        headers = [title] + [lang.LABEL for lang in languages.values()]
        tables.append(
            tabulate(
                table,
                headers=headers,
                tablefmt='presto',
                disable_numparse=True,
                colalign=['left'] + ['center'] * len(languages),
            )
        )
    legend = (
        'Templates: ᴀ attached, ᴄ combined, ᴇ exported, '
        'ᴾ also predict_proba, ··· not supported'
    )
    return '\n\n'.join([*tables, legend])


def _get_public_name(clazz: type) -> str:
    """
    Get the public name of a class of scikit-learn without its private
    modules, e.g. `tree.DecisionTreeClassifier` instead of
    `sklearn.tree._classes.DecisionTreeClassifier`.
    """
    public = []
    for part in clazz.__module__.split('.')[1:]:
        if part.startswith('_'):
            break
        public.append(part)
    return '.'.join([*public, clazz.__qualname__])


def can(
    estimator: BaseEstimator | type,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    method: str | enum.Method | None = None,
) -> bool:
    """
    Check the support of the given arguments.

    Parameters
    ----------
    estimator : BaseEstimator or estimator class.
        Set a fitted base estimator of scikit-learn.
    language : str
        Set the target programming language.
    template : str
        Set the kind of desired template.
    method : str
        Set the kind of template.

    Returns
    -------
    True by a supported combination.
    """
    return any(
        clazz.supports(
            estimator, language=language, template=template, method=method
        )
        for clazz in (Estimator, Transformer, Pipeline)
    )


def port(
    estimator: BaseEstimator,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    to_json: bool = False,
    ignore_preprocessing: bool = False,
) -> str | tuple[str, str]:
    return Porter(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
        ignore_preprocessing=ignore_preprocessing,
    ).port(to_json)


def save(
    estimator: BaseEstimator,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    directory: str | Path | None = None,
    to_json: bool = False,
    ignore_preprocessing: bool = False,
) -> str | tuple[str, str]:
    return Porter(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
        ignore_preprocessing=ignore_preprocessing,
    ).save(directory=directory, to_json=to_json)


def make(
    estimator: BaseEstimator,
    x: ArrayLike | spmatrix,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    n_jobs: bool | int = True,
    directory: str | Path | None = None,
    delete_created_files: bool = True,
    check_dependencies: bool = True,
    ignore_preprocessing: bool = False,
) -> tuple[Any, Any] | np.ndarray:
    return Porter(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
        ignore_preprocessing=ignore_preprocessing,
    ).make(
        x,
        n_jobs=n_jobs,
        directory=directory,
        delete_created_files=delete_created_files,
        check_dependencies=check_dependencies,
    )


def test(
    estimator: BaseEstimator,
    x: list | np.ndarray,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    n_jobs: bool | int = True,
    directory: str | Path | None = None,
    delete_created_files: bool = True,
    normalize: bool = True,
    ignore_preprocessing: bool = False,
) -> float:
    return Porter(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
        ignore_preprocessing=ignore_preprocessing,
    ).test(
        x,
        n_jobs=n_jobs,
        directory=directory,
        delete_created_files=delete_created_files,
        normalize=normalize,
    )
