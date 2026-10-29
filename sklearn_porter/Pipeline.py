from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.Base import Base, _get_qualname, _parse_predictions
from sklearn_porter.Estimator import Estimator, _integrity_score, _predictions
from sklearn_porter.pipeline.Pipeline import Pipeline as PortedPipeline
from sklearn_porter.Transformer import Transformer


def _get_pipeline(estimator: object) -> SklearnPipeline | None:
    """
    Get the pipeline of the passed object.

    Parameters
    ----------
    estimator : object
        A pipeline or a fitted `GridSearchCV` or `RandomizedSearchCV` with a
        pipeline as best estimator.

    Returns
    -------
    The pipeline or None.
    """
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
        estimator = getattr(estimator, 'best_estimator_', None)
    return estimator if isinstance(estimator, SklearnPipeline) else None


def _split_steps(
    pipeline: SklearnPipeline,
) -> tuple[list, BaseEstimator | None]:
    """
    Split the steps of a pipeline into the preprocessing steps and the final
    estimator.

    Parameters
    ----------
    pipeline : Pipeline
        A pipeline of scikit-learn.

    Returns
    -------
    The preprocessing steps as tuples of the name and the step (without the
    skipped steps, which are `None` or `'passthrough'`) and the final
    estimator (or None, if it's skipped).
    """
    preprocessing = [
        (name, step)
        for name, step in pipeline.steps[:-1]
        if step is not None and step != 'passthrough'
    ]
    final_est = pipeline.steps[-1][1]
    if final_est is None or final_est == 'passthrough':
        final_est = None
    return preprocessing, final_est


@decorator.aliased
class Pipeline(Base):
    """
    Main class which validates the passed pipeline of transformers (e.g. a
    `StandardScaler`) and a final estimator and ports it to one source file.

    The transformers are validated by `Transformer` and the final estimator
    by `Estimator`. The ported pipeline is the ported final estimator, which
    transforms the features of each sample with the ported transformers
    before its predictions, so it predicts like the original pipeline.
    """

    KIND = 'pipeline'

    def __init__(
        self,
        pipeline: BaseEstimator,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        class_name: str | None = None,
        converter: Callable[[object], str] | None = None,
    ):
        """
        Validate and coordinate the passed pipeline for transpiling.

        Parameters
        ----------
        pipeline : BaseEstimator
            Set a fitted pipeline of scikit-learn (also as best estimator of
            a `GridSearchCV` or `RandomizedSearchCV`).
        language : str or Language, optional
            The target programming language of all steps. By default the
            default language of the final estimator will be used.
        template : str or Template, optional
            The template of the generated code of all steps. By default the
            default template of the final estimator will be used.
        class_name : str
            The class name of the ported final estimator, which is also the
            name of the generated file, by default `Pipeline`. The ported
            transformers keep the class names of their estimators, e.g.
            `StandardScaler` (with a number for repeated names, e.g.
            `StandardScaler2`).
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.

        Raises
        ------
        NotSupportedYetError
            If a preprocessing step or the final estimator isn't supported.
        """
        self._init_environment()
        self.estimator = pipeline  # see @estimator.setter

        # Set defaults:
        self.converter = converter or str
        self.language = language or self._final.language
        self.template = template or self._final.template
        self.class_name = class_name or 'Pipeline'

    @property
    def _name(self) -> str:
        names = (step._name for step in self.steps)
        return f'Pipeline({", ".join(names)})'

    @property
    def estimator(self):
        return self._pipeline

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        self._pipeline = self._extract(estimator)
        preprocessing, final_est = _split_steps(self._pipeline)
        assert final_est is not None  # see `_extract()`
        self._transformers = [Transformer(step) for _, step in preprocessing]
        self._final = Estimator(final_est)

        # Unique class names of the ported transformers (e.g.
        # `StandardScaler` and `StandardScaler2`) in the generated file:
        class_names = set()
        for step in self._transformers:
            class_name, i = step.class_name, 1
            while class_name in class_names:
                i += 1
                class_name = f'{step.class_name}{i}'
            class_names.add(class_name)
            step.class_name = class_name

        self._estimator = PortedPipeline(
            self._pipeline, self._transformers, self._final
        )

    @property
    def steps(self) -> tuple[Transformer | Estimator, ...]:
        """The ported steps: the transformers and the final estimator."""
        return (*self._transformers, self._final)

    @Base.language.setter
    def language(self, language: str | enum.Language):
        language = enum.Language.convert(language)
        if not self.supports(self.estimator, language):
            msg = (
                f'The passed language `{language.value.KEY}` is not '
                f'supported for all steps of the pipeline `{self._name}`.'
            )
            raise exception.NotSupportedYetError(msg)
        for step in self.steps:
            step.language = language
        self._language = language

    @Base.template.setter
    def template(self, name: str | enum.Template):
        template = enum.Template.convert(name)
        if not self.supports(self.estimator, self.language, template):
            msg = (
                f'The passed template `{template.value}` is not '
                f'supported for all steps of the pipeline `{self._name}` '
                f'and language `{self.language}`.'
            )
            raise exception.NotSupportedYetError(msg)
        for step in self.steps:
            step.template = template
        self._template = template

    @Base.converter.setter
    def converter(self, converter: Callable[[object], str]):
        for step in self.steps:
            step.converter = converter
        self._converter = converter

    def _extract(self, estimator: BaseEstimator) -> SklearnPipeline:
        """
        Extract and validate the pipeline.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted pipeline of scikit-learn (also as best estimator of
            a `GridSearchCV` or `RandomizedSearchCV`).

        Returns
        -------
        A valid pipeline.
        """
        qualname = _get_qualname(estimator)
        L.debug(f'Start validation of the passed pipeline: `{qualname}`.')

        pipeline = _get_pipeline(estimator)
        if pipeline is None:
            msg = (
                f'The passed object of type `{qualname}` is not a pipeline '
                '(or a fitted `GridSearchCV` or `RandomizedSearchCV` with a '
                'pipeline as best estimator).'
            )
            L.error(msg)
            raise ValueError(msg)

        preprocessing, final_est = _split_steps(pipeline)
        if final_est is None:
            msg = 'There is no final estimator in the pipeline.'
            L.error(msg)
            raise ValueError(msg)
        if not preprocessing:
            msg = (
                'The pipeline has no preprocessing steps. Port its final '
                'estimator with `Estimator` or `Porter`.'
            )
            L.error(msg)
            raise ValueError(msg)

        unsupported = [
            f'`{name}` ({step.__class__.__name__})'
            for name, step in preprocessing
            if not Transformer.supports_step(step)
        ]
        if unsupported:
            kind = 'steps' if len(unsupported) > 1 else 'step'
            msg = (
                f'The pipeline contains the preprocessing {kind} '
                f"{', '.join(unsupported)}, which can't be ported yet."
            )
            L.error(msg)
            raise exception.NotSupportedYetError(msg)

        return pipeline

    @classmethod
    def supports(
        cls,
        estimator: BaseEstimator | type,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        method: str | enum.Method | None = None,
    ) -> bool:
        """
        Check the support of the given arguments for all steps of a pipeline.

        Parameters
        ----------
        estimator : BaseEstimator or estimator class.
            Set a fitted pipeline of scikit-learn (also as best estimator of
            a `GridSearchCV` or `RandomizedSearchCV`).
        language : str
            Set the target programming language.
        template : str
            Set the kind of desired template.
        method : str
            Set the method of the final estimator, e.g. `predict_proba`.

        Returns
        -------
        True, if the pipeline has preprocessing steps and all steps support
        the given combination.
        """
        pipeline = _get_pipeline(estimator)
        if pipeline is None:
            return False
        preprocessing, final_est = _split_steps(pipeline)
        if not preprocessing or final_est is None:
            return False
        if language:
            support = PortedPipeline.SUPPORT.get(
                enum.Language.convert(language), {}
            )
            if template and enum.Template.convert(template) not in support:
                return False
            if not support:
                return False
        tr_method = enum.Method.TRANSFORM if method else None
        return all(
            Transformer.supports_step(step, language, template, tr_method)
            for _, step in preprocessing
        ) and Estimator.supports(final_est, language, template, method)

    @decorator.alias('predict')
    def make(
        self,
        x: list | np.ndarray,
        n_jobs: bool | int = True,
        directory: str | Path | None = None,
        delete_created_files: bool = True,
        check_dependencies: bool = True,
        **kwargs,
    ) -> (
        tuple[np.int64, np.ndarray | None]
        | tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]
    ):
        """
        Make predictions with the transpiled pipeline locally.

        Parameters
        ----------
        x : list or np.ndarray of shape (n_samples, n_features) or (n_features)
            Input data.
        n_jobs : bool or int (default: True, which uses `os.cpu_count()`)
            The maximal number of processes to make the predictions. Each
            process predicts a batch of at least 100 samples (or all
            samples, if there are fewer).
        directory : str or Path, optional (default: a temporary directory)
            Set the directory where all generated files should be saved.
        delete_created_files : bool (default: True)
            Whether to delete the generated files finally or not.
        check_dependencies : bool (default: True)
            Check whether all required applications are in the PATH or not.
        kwargs

        Returns
        -------
        Return the predictions and probabilities of the final estimator.
        """
        y = self._run(
            x,
            n_jobs=n_jobs,
            directory=directory,
            delete_created_files=delete_created_files,
            check_dependencies=check_dependencies,
            parse=_parse_predictions,
            **kwargs,
        )
        return _predictions(y, self._final._estimator)

    @decorator.alias('integrity_score')
    def test(
        self,
        x,
        n_jobs: bool | int = True,
        directory: str | Path | None = None,
        delete_created_files: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> float:
        """
        Compute the integrity score of the ported pipeline.

        The score is the share of predictions of the ported pipeline which
        are equal (classifiers) or close (regressors) to the predictions of
        the original pipeline.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Input data.
        n_jobs : bool or int (default: True, which uses `os.cpu_count()`)
            The maximal number of processes to make the predictions. Each
            process predicts a batch of at least 100 samples (or all
            samples, if there are fewer).
        directory : str or Path, optional (default: a temporary directory)
            Set the directory where all generated files should be saved.
        delete_created_files : bool (default: True)
            Whether to delete the generated files finally or not.
        normalize : bool, default: True
            Whether to normalize the result or not.

        Returns
        -------
        score : float
            Return the relative (or with `normalize=False` the absolute)
            number of equal predictions.
        """
        self._check_kwargs(kwargs)
        y_true = self._pipeline.predict(x)
        y_pred = self.make(
            x,
            n_jobs=n_jobs,
            directory=directory,
            delete_created_files=delete_created_files,
        )[0]  # only predicts
        return _integrity_score(
            self._final.estimator, y_true, y_pred, normalize
        )
