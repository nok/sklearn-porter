import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

# Additional installed modules:
import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaseEnsemble,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Perceptron,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import (
    SVC,
    SVR,
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.Base import Base, _get_qualname, _parse_predictions
from sklearn_porter.Transformer import Transformer

if TYPE_CHECKING:
    from sklearn_porter.estimator.EstimatorBase import EstimatorBase


@decorator.aliased
class Estimator(Base):
    """
    Main class which validates the passed estimator (a classifier, a
    regressor or an outlier detector) and coordinates the kind of estimator
    to a concrete subclass.
    """

    KIND = 'estimator'

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
        Validate and coordinate the passed estimator for transpiling.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.
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
            steps (e.g. a `StandardScaler`). By default such a pipeline raises
            a `NotSupportedYetError`, because the predictions of the ported
            estimator would differ from those of the pipeline. If True, the
            preprocessing steps are ignored with a `UserWarning`, so apply
            them to the input data of the ported estimator yourself (also to
            the samples of `make()` and `test()`, e.g. with
            `pipeline[:-1].transform(x)`).

        Raises
        ------
        NotSupportedYetError
            If the passed pipeline contains preprocessing steps and
            `ignore_preprocessing` is False.
        """
        self.ignore_preprocessing = ignore_preprocessing
        super().__init__(
            estimator,
            language=language,
            template=template,
            class_name=class_name,
            converter=converter,
        )

    def _extract(self, estimator: BaseEstimator) -> BaseEstimator | None:
        return self._extract_est(estimator, self.ignore_preprocessing)

    @staticmethod
    def _extract_est(
        estimator: BaseEstimator, ignore_preprocessing: bool = False
    ) -> BaseEstimator | None:
        """
        Extract the original estimator.

        Check if the estimator is a valid base estimator of scikit-learn.
        Check if the estimator is embedded in an optimizer or pipeline.
        Check if the estimator is a classifier, regressor or outlier detector.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.
        ignore_preprocessing : bool, default=False
            Ignore the preprocessing steps of a pipeline with a warning
            instead of raising an error.

        Returns
        -------
        A valid base estimator or None.
        """
        est = estimator  # shorter <3
        qualname = _get_qualname(est)

        L.debug(f'Start validation of the passed estimator: `{qualname}`.')

        # Check BaseEstimator:
        if not isinstance(est, BaseEstimator):
            msg = (
                f'The passed estimator `{qualname}` is not a '
                f'valid base estimator of scikit-learn v{sklearn_version} .'
            )
            L.error(msg)
            raise ValueError(msg)

        # Check BaseEnsemble:
        if isinstance(est, BaseEnsemble):
            # A list or an array (e.g. of the stages of gradient boosting):
            estimators = getattr(est, 'estimators_', None)
            if estimators is None or len(estimators) == 0:
                raise exception.NotFittedEstimatorError(qualname)

        # Check GridSearchCV and RandomizedSearchCV:
        L.debug('Check whether the estimator is embedded in an optimizer.')
        if isinstance(est, BaseSearchCV):
            L.debug('└> Yes, the estimator is embedded in an optimizer.')
            if not isinstance(est, (GridSearchCV, RandomizedSearchCV)):
                msg = (
                    f'The used optimizer `{qualname}` is not supported '
                    'by this version of sklearn-porter. Try to '
                    'extract the internal estimator manually '
                    'and pass it.'
                )
                L.error(msg)
                raise ValueError(msg)
            if getattr(est, 'best_estimator_', None) is None:
                msg = 'The embedded estimator is not fitted.'
                L.error(msg)
                raise ValueError(msg)
            est = est.best_estimator_
            msg = (
                f'Extract the embedded estimator of type `{_get_qualname(est)}` '
                f'from optimizer `{qualname}`.'
            )
            L.info(msg)
        else:
            L.debug('└> No, the estimator is not embedded in an optimizer.')

        # Check Pipeline:
        L.debug('Check whether the estimator is embedded in a pipeline.')
        if isinstance(est, Pipeline):
            L.debug('└> Yes, the estimator is embedded in a pipeline.')
            final_est = est._final_estimator
            if final_est is None or final_est == 'passthrough':
                msg = 'There is no final estimator in the pipeline.'
                L.error(msg)
                raise ValueError(msg)
            # The preprocessing steps (e.g. a `StandardScaler`) can't be
            # ported, so the predictions of the final estimator alone would
            # differ from those of the pipeline:
            preprocessing = [
                (name, step)
                for name, step in est.steps[:-1]
                if step is not None and step != 'passthrough'
            ]
            steps = [
                f'`{name}` ({step.__class__.__name__})'
                for name, step in preprocessing
            ]
            if steps:
                plural = len(steps) > 1
                names = ', '.join(steps)
                kind = 'steps' if plural else 'step'
                if not ignore_preprocessing:
                    msg = (
                        f'The pipeline contains the preprocessing {kind} '
                        f"{names}, which can't be ported. Only the final "
                        f'estimator `{final_est.__class__.__name__}` would be '
                        'ported, so its predictions would differ from those '
                        'of the pipeline. Pass `ignore_preprocessing=True` '
                        'to port the final estimator anyway and apply the '
                        'preprocessing to its input data yourself.'
                    )
                    # All steps can be ported as transformers:
                    if all(Transformer.supports(s) for _, s in preprocessing):
                        msg += ' Or port the whole pipeline with `Porter`.'
                    L.error(msg)
                    raise exception.NotSupportedYetError(msg)
                them = 'them' if plural else 'it'
                warnings.warn(
                    f'The preprocessing {kind} {names} of the pipeline '
                    f'{"are" if plural else "is"} ignored, so apply {them} '
                    'to the input data of the ported estimator yourself.',
                    UserWarning,
                    stacklevel=4,
                )
            est = final_est
            msg = (
                'Extract the embedded estimator of type '
                f'`{_get_qualname(est)}` from the pipeline.'
            )
            L.info(msg)
        else:
            L.debug('└> No, the estimator is not embedded in a pipeline.')

        # Check ClassifierMixin:
        L.debug(
            'Check whether the estimator is inherited from `ClassifierMixin`.'
        )
        is_classifier = isinstance(est, ClassifierMixin)
        if is_classifier:
            L.debug(
                '└> Yes, the estimator is inherited from `ClassifierMixin`.'
            )
            return est
        L.debug('└> No, the estimator is not inherited from `ClassifierMixin`.')

        # Check RegressorMixin:
        L.debug(
            'Check whether the estimator is inherited from `RegressorMixin`.'
        )
        is_regressor = isinstance(est, RegressorMixin)
        if is_regressor:
            L.debug('└> Yes, the estimator is inherited from `RegressorMixin`.')
            return est
        L.debug('└> No, the estimator is not inherited from `RegressorMixin`.')

        # Check OutlierMixin:
        L.debug('Check whether the estimator is inherited from `OutlierMixin`.')
        is_outlier_detector = isinstance(est, OutlierMixin)
        if is_outlier_detector:
            L.debug('└> Yes, the estimator is inherited from `OutlierMixin`.')
            return est
        L.debug('└> No, the estimator is not inherited from `OutlierMixin`.')

        if not (is_classifier or is_regressor or is_outlier_detector):
            msg = (
                f'The passed object of type `{qualname}` is neither '
                'a classifier, a regressor nor an outlier detector.'
            )
            if isinstance(est, TransformerMixin):
                msg += ' Port a transformer with `Transformer` or `Porter`.'
            L.error(msg)
            raise ValueError(msg)

        return None

    @staticmethod
    def _registry() -> tuple:
        return Estimator.estimators()

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
        Make predictions with transpiled estimators locally.

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
        Return the predictions and probabilities.

        Warns
        -----
        UserWarning
            On Windows, because the compilation and execution of the ported
            estimators isn't tested on Windows.
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
        return _predictions(y, self._estimator)

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
        Compute the integrity score of the ported estimator.

        The score is the share of predictions of the ported estimator which
        are equal (classifiers) or close (regressors) to the predictions of
        the original estimator.

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
        y_true = self._estimator.estimator.predict(x)
        y_pred = self.make(
            x,
            language=self._language,
            template=self._template,
            n_jobs=n_jobs,
            directory=directory,
            delete_created_files=delete_created_files,
        )
        y_pred = y_pred[0]  # only predicts
        return _integrity_score(
            self._estimator.estimator, y_true, y_pred, normalize
        )

    @staticmethod
    def classifiers() -> tuple:
        """
        Get a set of supported and installed classifiers.

        Returns
        -------
        estimators : tuple
            A set of supported classifiers.
        """
        return (
            AdaBoostClassifier,
            BernoulliNB,
            ComplementNB,
            DecisionTreeClassifier,
            ExtraTreesClassifier,
            GaussianNB,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            KNeighborsClassifier,
            LinearSVC,
            LogisticRegression,
            LogisticRegressionCV,
            MLPClassifier,
            MultinomialNB,
            NuSVC,
            Perceptron,
            RandomForestClassifier,
            RidgeClassifier,
            RidgeClassifierCV,
            SGDClassifier,
            SVC,
        )

    @staticmethod
    def estimators() -> tuple:
        """
        Get a set of all supported and installed estimators.

        Returns
        -------
        estimators : tuple
            A set of supported classifiers, regressors and outlier detectors.
        """
        return (
            Estimator.classifiers()
            + Estimator.regressors()
            + Estimator.outlier_detectors()
        )

    @staticmethod
    def outlier_detectors() -> tuple:
        """
        Get a set of supported and installed outlier detectors.

        Returns
        -------
        estimators : tuple
            A set of supported outlier detectors.
        """
        return (IsolationForest, OneClassSVM)

    @staticmethod
    def regressors() -> tuple:
        """
        Get a set of supported and installed regressors.

        Returns
        -------
        estimators : tuple
            A set of supported regressors.
        """
        return (
            ARDRegression,
            BayesianRidge,
            DecisionTreeRegressor,
            ElasticNet,
            ElasticNetCV,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            HuberRegressor,
            KNeighborsRegressor,
            Lasso,
            LassoCV,
            LinearRegression,
            LinearSVR,
            MLPRegressor,
            NuSVR,
            RandomForestRegressor,
            Ridge,
            RidgeCV,
            SGDRegressor,
            SVR,
        )


def _integrity_score(
    estimator: BaseEstimator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compare the predictions of the ported and the original estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The original estimator (e.g. the final estimator of a pipeline).
    y_true : np.ndarray
        The predictions of scikit-learn.
    y_pred : np.ndarray
        The predictions of the ported estimator.
    normalize : bool, default: True
        Whether to normalize the result or not.

    Returns
    -------
    score : float
        The relative (or with `normalize=False` the absolute) number of
        equal (classifiers) or close (regressors) predictions.
    """
    if isinstance(estimator, RegressorMixin):
        # Small deviations because of the floating point arithmetic:
        is_close = np.isclose(y_true, y_pred)
        if is_close.ndim > 1:  # multiple outputs
            is_close = is_close.all(axis=1)
        score = is_close.mean() if normalize else is_close.sum()
        return float(score)
    if np.ndim(y_true) > 1:  # multiple outputs or labels
        is_equal = (y_true == np.reshape(y_pred, np.shape(y_true))).all(axis=1)
        return float(is_equal.mean() if normalize else is_equal.sum())
    return float(accuracy_score(y_true, y_pred, normalize=normalize))


def _predictions(
    y: list, ported_est: 'EstimatorBase'
) -> (
    tuple[np.int64, np.ndarray | None]
    | tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]
):
    """
    Convert the results of a ported estimator to predictions.

    Parameters
    ----------
    y : list
        The results (the prediction and the probabilities) of each sample.
    ported_est : EstimatorBase
        The ported estimator (e.g. the final estimator of a pipeline).

    Returns
    -------
    Return the predictions and probabilities.
    """
    y = list(zip(*y, strict=False))
    y_pred = np.array(y[0])
    meta_info = ported_est.meta_info
    if 'labels' in meta_info and not meta_info.get('is_multilabel'):
        # The ported classifiers return the labels of the classes, so
        # convert them (e.g. `1` of JSON to `1.0`) to the type of the
        # labels of scikit-learn:
        classes = ported_est.estimator.classes_  # ty: ignore[unresolved-attribute]
        if not isinstance(classes, list):
            classes = [classes]
        y_pred = y_pred.astype(np.result_type(*classes))
    y_proba: np.ndarray | list[np.ndarray] | None = None
    if len(y) > 1:
        if ported_est.meta_info.get('is_multi_output'):
            # Multi-output classifiers return the probabilities of each
            # output (with different numbers of classes), so create one
            # array per output like scikit-learn:
            n_outputs = len(y[1][0])
            y_proba = [np.array([p[i] for p in y[1]]) for i in range(n_outputs)]
        else:
            y_proba = np.array(y[1])

    if len(y_pred) == 1:  # a single sample
        if isinstance(y_proba, list):  # multiple outputs
            return y_pred[0], [p[0] for p in y_proba]
        return y_pred[0], None if y_proba is None else y_proba[0]
    return y_pred, y_proba
