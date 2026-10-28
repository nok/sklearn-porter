from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger as L
from scipy.sparse import issparse

# scikit-learn
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.Base import Base, _get_qualname, _parse_values


def is_transformer(estimator: object) -> bool:
    """
    Check whether the passed object is a transformer.

    Parameters
    ----------
    estimator : object
        The passed object, e.g. a fitted estimator of scikit-learn.

    Returns
    -------
    True if the object is a transformer (e.g. a `StandardScaler`), but not a
    classifier, a regressor or an outlier detector.
    """
    return isinstance(estimator, TransformerMixin) and not isinstance(
        estimator, (ClassifierMixin, RegressorMixin, OutlierMixin)
    )


@decorator.aliased
class Transformer(Base):
    """
    Main class which validates the passed transformer (e.g. a
    `StandardScaler`) and coordinates the kind of transformer to a concrete
    subclass.

    The ported transformer has the method `transform`, which returns the
    transformed features of one sample. Its `main` prints the transformed
    features of each sample in one line (separated by spaces), so the output
    can be passed to a ported estimator.
    """

    KIND = 'transformer'

    def __init__(
        self,
        transformer: BaseEstimator,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        class_name: str | None = None,
        converter: Callable[[object], str] | None = None,
    ):
        """
        Validate and coordinate the passed transformer for transpiling.

        Parameters
        ----------
        transformer : BaseEstimator
            Set a fitted transformer of scikit-learn, e.g. a
            `StandardScaler`.
        language : str or Language, optional
            The target programming language. By default the default language
            of the transformer will be used.
        template : str or Template, optional
            The template of the generated code. By default the default
            template of the transformer will be used.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed transformer will
            be used, e.g. `StandardScaler`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        """
        super().__init__(
            transformer,
            language=language,
            template=template,
            class_name=class_name,
            converter=converter,
        )

    @property
    def transformer(self):
        return self.estimator

    def _extract(self, estimator: BaseEstimator) -> BaseEstimator | None:
        return self._extract_est(estimator)

    @staticmethod
    def _extract_est(estimator: BaseEstimator) -> BaseEstimator:
        """
        Extract the original transformer.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted transformer of scikit-learn.

        Returns
        -------
        A valid transformer.
        """
        est = estimator  # shorter <3
        qualname = _get_qualname(est)

        L.debug(f'Start validation of the passed transformer: `{qualname}`.')

        if not isinstance(est, BaseEstimator):
            msg = (
                f'The passed transformer `{qualname}` is not a '
                'valid base estimator of scikit-learn.'
            )
            L.error(msg)
            raise ValueError(msg)

        if isinstance(est, Pipeline):
            msg = (
                "A pipeline can't be ported as transformer. Port it with "
                '`Porter` (or `Pipeline`).'
            )
            L.error(msg)
            raise exception.NotSupportedYetError(msg)

        if not is_transformer(est):
            msg = (
                f'The passed object of type `{qualname}` is not a '
                'transformer. Port a classifier, a regressor or an outlier '
                'detector with `Estimator` or `Porter`.'
            )
            L.error(msg)
            raise ValueError(msg)

        return est

    @staticmethod
    def _registry() -> tuple:
        return Transformer.transformers()

    def make(
        self,
        x: list | np.ndarray,
        n_jobs: bool | int = True,
        directory: str | Path | None = None,
        delete_created_files: bool = True,
        check_dependencies: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Transform samples with the transpiled transformer locally.

        Parameters
        ----------
        x : list or np.ndarray of shape (n_samples, n_features) or (n_features)
            Input data.
        n_jobs : bool or int (default: True, which uses `os.cpu_count()`)
            The maximal number of processes to transform the samples. Each
            process transforms a batch of at least 100 samples (or all
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
        The transformed features of shape (n_samples, n_features_out), or of
        shape (n_features_out) for a single sample of shape (n_features).

        Warns
        -----
        UserWarning
            On Windows, because the compilation and execution of the ported
            transformers isn't tested on Windows.
        """
        is_single = not issparse(x) and np.ndim(x) == 1
        values = self._run(
            x,
            n_jobs=n_jobs,
            directory=directory,
            delete_created_files=delete_created_files,
            check_dependencies=check_dependencies,
            parse=_parse_values,
            **kwargs,
        )
        x_tr = np.array(values, dtype=np.float64)
        return x_tr[0] if is_single else x_tr

    # Transform the samples like a transformer of scikit-learn:
    transform = make

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
        Compute the integrity score of the ported transformer.

        The score is the share of samples whose features transformed by the
        ported transformer are close to the features transformed by the
        original transformer (missing values are equal).

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Input data.
        n_jobs : bool or int (default: True, which uses `os.cpu_count()`)
            The maximal number of processes to transform the samples. Each
            process transforms a batch of at least 100 samples (or all
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
            number of samples with equal transformed features.
        """
        self._check_kwargs(kwargs)
        if not issparse(x) and np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))
        x_true = self._estimator.estimator.transform(x)
        if issparse(x_true):
            x_true = x_true.toarray()
        x_tr = self.make(
            x,
            language=self._language,
            template=self._template,
            n_jobs=n_jobs,
            directory=directory,
            delete_created_files=delete_created_files,
        )
        is_close = np.isclose(x_true, x_tr, equal_nan=True).all(axis=1)
        return float(is_close.mean() if normalize else is_close.sum())

    @staticmethod
    def transformers() -> tuple:
        """
        Get a set of supported and installed transformers.

        Returns
        -------
        transformers : tuple
            A set of supported transformers.
        """
        return (StandardScaler,)
