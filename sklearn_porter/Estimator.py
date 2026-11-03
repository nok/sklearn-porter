import warnings
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Additional installed modules:
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse, spmatrix

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
from sklearn.exceptions import NotFittedError
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
from sklearn.utils.validation import check_is_fitted

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.Base import Base, _get_qualname, _parse_predictions
from sklearn_porter.Transformer import Transformer

L = getLogger(__name__)

if TYPE_CHECKING:
    from sklearn_porter.estimator.EstimatorBase import EstimatorBase


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

        # Check BaseEstimator:
        if not isinstance(est, BaseEstimator):
            msg = (
                f'The passed estimator `{qualname}` is not a '
                f'valid base estimator of scikit-learn v{sklearn_version} .'
            )
            raise ValueError(msg)

        # Check whether the estimator (or the optimizer or pipeline) is fitted:
        try:
            check_is_fitted(est)
        except NotFittedError:
            raise exception.NotFittedEstimatorError(qualname) from None

        # Check GridSearchCV and RandomizedSearchCV:
        if isinstance(est, (GridSearchCV, RandomizedSearchCV)):
            if getattr(est, 'best_estimator_', None) is None:
                msg = 'The embedded estimator is not fitted.'
                raise ValueError(msg)
            est = est.best_estimator_
            msg = (
                f'Extract the embedded estimator of type `{_get_qualname(est)}` '
                f'from optimizer `{qualname}`.'
            )
            L.debug(msg)

        # Check Pipeline:
        if isinstance(est, Pipeline):
            final_est = est.steps[-1][1]
            if final_est is None or final_est == 'passthrough':
                msg = 'There is no final estimator in the pipeline.'
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
                    if all(
                        Transformer.supports_step(s) for _, s in preprocessing
                    ):
                        msg += ' Or port the whole pipeline with `Porter`.'
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
            L.debug(msg)

        # Check ClassifierMixin:
        is_classifier = isinstance(est, ClassifierMixin)
        if is_classifier:
            return est

        # Check RegressorMixin:
        is_regressor = isinstance(est, RegressorMixin)
        if is_regressor:
            return est

        # Check OutlierMixin:
        is_outlier_detector = isinstance(est, OutlierMixin)
        if is_outlier_detector:
            return est

        if not (is_classifier or is_regressor or is_outlier_detector):
            msg = (
                f'The passed object of type `{qualname}` is neither '
                'a classifier, a regressor nor an outlier detector.'
            )
            if isinstance(est, TransformerMixin):
                msg += ' Port a transformer with `Transformer` or `Porter`.'
            raise ValueError(msg)

        return None

    @staticmethod
    def _registry() -> tuple:
        return Estimator.estimators()

    def make(
        self,
        x: ArrayLike | spmatrix,
        n_jobs: bool | int = True,
        directory: str | Path | None = None,
        delete_created_files: bool = True,
        check_dependencies: bool = True,
        **kwargs,
    ) -> tuple[Any, Any]:
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
        Return the predictions and probabilities, of a single sample
        of shape (n_features) without the dimension of the samples.

        Warns
        -----
        UserWarning
            On Windows, because the compilation and execution of the ported
            estimators isn't tested on Windows.
        """
        with self._overrides(kwargs):
            y = self._run(
                x,
                n_jobs=n_jobs,
                directory=directory,
                delete_created_files=delete_created_files,
                check_dependencies=check_dependencies,
                parse=_parse_predictions,
            )
        return _predictions(y, self._estimator, _is_single(x))

    # Make predictions like an estimator of scikit-learn:
    predict = make

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
        x = _as_samples(x)
        y_true = self._estimator.estimator.predict(x)
        with self._overrides(kwargs):
            y_pred = self.make(
                x,
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


def _is_single(x) -> bool:
    """Check whether the passed samples are a single sample."""
    return not issparse(x) and np.ndim(x) == 1


def _as_samples(x):
    """Reshape a single sample to samples (like scikit-learn expects)."""
    return np.reshape(x, (1, -1)) if _is_single(x) else x


def _predictions(
    y: list, ported_est: 'EstimatorBase', is_single: bool = False
) -> tuple[Any, Any]:
    """
    Convert the results of a ported estimator to predictions.

    Parameters
    ----------
    y : list
        The results (the prediction and the probabilities) of each sample.
    ported_est : EstimatorBase
        The ported estimator (e.g. the final estimator of a pipeline).
    is_single : bool (default: False)
        Whether the passed samples were a single sample of shape
        (n_features), whose prediction and probabilities are returned
        without the dimension of the samples (like a transformer does).

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
        # Floating point numbers, also if JSON contains only integers
        # (e.g. `1` instead of `1.0` of JavaScript):
        if ported_est.meta_info.get('is_multi_output'):
            # Multi-output classifiers return the probabilities of each
            # output (with different numbers of classes), so create one
            # array per output like scikit-learn:
            n_outputs = len(y[1][0])
            y_proba = [
                np.array([p[i] for p in y[1]], dtype=np.float64)
                for i in range(n_outputs)
            ]
        else:
            y_proba = np.array(y[1], dtype=np.float64)

    if is_single:
        if isinstance(y_proba, list):  # multiple outputs
            return y_pred[0], [p[0] for p in y_proba]
        return y_pred[0], None if y_proba is None else y_proba[0]
    return y_pred, y_proba
