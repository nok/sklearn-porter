import urllib.request
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from json import JSONDecodeError, loads
from os import cpu_count, environ, pathsep, remove
from pathlib import Path
from shlex import join
from shutil import rmtree, which
from subprocess import STDOUT, CalledProcessError, check_output, run
from sys import platform, stdout, version_info
from tempfile import mkdtemp
from textwrap import dedent
from time import sleep

# Additional installed modules:
import numpy as np
from loguru import logger as L
from scipy.sparse import issparse

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    OutlierMixin,
    RegressorMixin,
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
from tabulate import tabulate

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter import meta
from sklearn_porter.utils import densify, options, tabify

# The minimal number of samples which `make()` passes to one process of the
# ported estimator, because starting a process takes much longer than a
# prediction:
MIN_BATCH_SIZE = 100


@decorator.aliased
class Estimator:
    """
    Main class which validates the passed estimator and
    coordinates the kind of estimator to a concrete subclass.
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
        Validate and coordinate the passed estimator for transpiling.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `DecisionTreeClassifier`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        ignore_preprocessing : bool, default=False
            Port only the final estimator of a pipeline with preprocessing
            steps (e.g. a `StandardScaler`), which can't be ported. By default
            such a pipeline raises a `NotSupportedYetError`, because the
            predictions of the ported estimator would differ from those of
            the pipeline. If True, the preprocessing steps are ignored with a
            `UserWarning`, so apply them to the input data of the ported
            estimator yourself (also to the samples of `make()` and `test()`,
            e.g. with `pipeline[:-1].transform(x)`).

        Raises
        ------
        NotSupportedYetError
            If the passed pipeline contains preprocessing steps and
            `ignore_preprocessing` is False.
        """
        L.remove()  # remove basic logger
        L.add(stdout, level=options['logging.level'])

        self.python_version = '.'.join(map(str, version_info[:3]))
        self.porter_version = str(meta.__version__)

        L.debug(f'Platform: {platform}')
        L.debug(f'Python: v{self.python_version}')
        L.debug(f'Package: scikit-learn: v{sklearn_version}')
        L.debug(f'Package: sklearn-porter: v{self.porter_version}')

        self.ignore_preprocessing = ignore_preprocessing
        self.estimator = estimator  # see @estimator.setter

        # Set defaults:
        self.language = language or self._estimator.DEFAULT_LANGUAGE
        self.template = template or self._estimator.DEFAULT_TEMPLATE
        self.class_name = class_name or self._estimator.estimator_name
        self.converter = converter or str

    @property
    def estimator(self):
        return self._estimator.estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        orig_est = self._extract_est(estimator, self.ignore_preprocessing)
        self._estimator = self._load_est(orig_est)

    @property
    def language(self) -> str:
        return self._language.value.KEY

    @language.setter
    def language(self, language: str | enum.Language):
        language = enum.Language.convert(language)
        if can(self.estimator, language):
            self._language = language
        else:
            name = self._estimator.estimator_name
            msg = (
                f'The passed language `{language.value.KEY}` is not '
                f'supported for the estimator `{name}`.'
            )
            raise exception.NotSupportedYetError(msg)

    @property
    def template(self) -> str:
        return self._template.value

    @template.setter
    def template(self, name: str | enum.Template):
        template = enum.Template.convert(name)
        if can(self.estimator, self.language, template):
            self._template = template
        else:
            name = self._estimator.estimator_name
            language = enum.Language.convert(self.language)
            msg = (
                f'The passed template `{template.value}` is not '
                f'supported for the estimator `{name}` '
                f'and language `{language.value.KEY}`.'
            )
            raise exception.NotSupportedYetError(msg)

    @property
    def class_name(self) -> str:
        return self._class_name

    @class_name.setter
    def class_name(self, class_name: str):
        self._class_name = class_name

    @property
    def converter(self) -> Callable[[object], str]:
        return self._converter

    @converter.setter
    def converter(self, converter: Callable[[object], str]):
        self._converter = converter

    def _check_kwargs(self, kwargs: dict) -> dict:
        """
        Check and save optional kwargs arguments internally.

        Parameters
        ----------
        kwargs : dict
            The additional passed arguments.

        Returns
        -------
        The original passed arguments.
        """
        if 'language' in kwargs:
            self.language = kwargs['language']
        if 'template' in kwargs:
            self.template = kwargs['template']
        class_name = kwargs.get('class_name')
        if isinstance(class_name, str) and len(class_name) > 0:
            self.class_name = class_name
        if 'converter' in kwargs:
            self.converter = kwargs['converter']
        return kwargs

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
            steps = [
                f'`{name}` ({step.__class__.__name__})'
                for name, step in est.steps[:-1]
                if step is not None and step != 'passthrough'
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
            L.error(msg)
            raise ValueError(msg)

        return None

    @staticmethod
    def _load_est(estimator):
        """
        Load the right subclass to read the passed estimator.

        Parameters
        ----------
        estimator : ClassifierMixin, RegressorMixin or OutlierMixin
            Set a fitted base estimator of scikit-learn.

        Returns
        -------
        A subclass from `sklearn_porter.estimator.*` which
        represents and includes the original base estimator.
        """
        est = estimator  # shorter <3
        qualname = _get_qualname(est)
        L.debug(f'Start loading the passed estimator: `{qualname}`.')

        name = est.__class__.__qualname__
        clazzes = Estimator.estimators()
        clazzes = {clazz.__name__: clazz for clazz in clazzes}
        if name in clazzes and isinstance(est, clazzes[name]):
            module = import_module(f'sklearn_porter.estimator.{name}')
            # Read the model data from dense arrays, because the ported
            # estimators don't support sparse matrices:
            ported_est = getattr(module, name)(densify(est))
            # Keep the original estimator for the predictions of scikit-learn
            # (e.g. SVMs fitted on sparse data need sparse support vectors):
            ported_est.estimator = est
            return ported_est

        msg = f'The passed estimator `{name}` is not supported.'
        raise exception.NotSupportedYetError(msg)

    def can(
        self,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        method: str | enum.Method | None = None,
    ) -> bool:
        return can(
            self.estimator, language=language, template=template, method=method
        )

    @decorator.alias('export')
    def port(self, to_json: bool = False, **kwargs) -> str | tuple[str, str]:
        """
        Port or transpile a passed estimator to a target programming language.

        Parameters
        ----------
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The transpiled estimator in the target programming language.
        """
        self._check_kwargs(kwargs)
        out = self._estimator.port(
            language=self._language,
            template=self._template,
            class_name=self.class_name,
            converter=self.converter,
            to_json=to_json,
        )
        if self._template == enum.Template.COMBINED and isinstance(out, str):
            out = tabify(out)
        return out

    @decorator.alias('dump')
    def save(
        self,
        directory: str | Path | None = None,
        to_json: bool = False,
        **kwargs,
    ) -> str | tuple[str, str]:
        """
        Port a passed estimator to a target programming language and save them.

        Parameters
        ----------
        directory : str or Path, optional (default: current working dir)
            Set the directory where all generated files should be saved.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The path(s) to the generated file(s).
        """
        self._check_kwargs(kwargs)
        return self._estimator.save(
            language=self._language,
            template=self._template,
            class_name=self.class_name,
            converter=self.converter,
            directory=directory,
            to_json=to_json,
        )

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
        self._check_kwargs(kwargs)
        if _is_windows():
            warnings.warn(
                'The compilation and execution of the ported estimators '
                "with `make()` and `test()` isn't tested on Windows.",
                UserWarning,
                stacklevel=2,
            )
        if check_dependencies:
            self._check_dependencies(self._language)

        temp_dir = None
        if not directory:
            directory = temp_dir = mkdtemp(prefix='sklearn-porter-')
        created_files = []  # for final deletion

        # Transpile model:
        out = self.save(directory=directory, to_json=True, **kwargs)

        if isinstance(out, tuple):  # indicator for Template.EXPORTED
            src_path, data_path = out[0], out[1]
            if not isinstance(data_path, Path):
                data_path = Path(data_path)
            data_path = data_path.resolve()
            created_files.append(data_path)
        else:
            src_path, data_path = out, None
        if not isinstance(src_path, Path):
            src_path = Path(src_path)
        src_path = src_path.resolve()

        created_files.append(src_path)
        class_paths = []

        language = self._language
        template = self._template

        # Compilation:
        self._compile(src_path, class_paths, created_files, language, template)

        # Execution:
        cmd_args = {}

        paths_to_check = []
        if language in (enum.Language.C, enum.Language.GO):
            dest_path = str(_get_binary_path(src_path))
            cmd_args['dest_path'] = dest_path
            paths_to_check.append(dest_path)
        elif language is enum.Language.CS:
            dest_path = str(src_path.with_suffix('.dll'))
            cmd_args['dest_path'] = dest_path
            paths_to_check.append(dest_path)
        elif language is enum.Language.JAVA:
            cmd_args['class_path'] = pathsep.join(class_paths)
            cmd_args['class_name'] = src_path.stem
        elif language in (
            enum.Language.JS,
            enum.Language.PHP,
            enum.Language.RUBY,
        ):
            cmd_args['src_path'] = str(src_path)
            paths_to_check.append(str(src_path))

        for path_to_check in paths_to_check:
            for _ in range(10):
                try:
                    fp = open(path_to_check)
                except OSError:
                    sleep(0.1)
                else:
                    fp.close()
                    break

        cmd = [arg.format(**cmd_args) for arg in language.value.CMD_EXECUTE]
        L.info(f'Execution command: `{join(cmd)}`')

        # Model data:
        if data_path:
            cmd.append(str(data_path))

        # Features:
        if issparse(x):
            x = x.toarray()  # ty: ignore[unresolved-attribute]
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        x = x.tolist()

        if isinstance(n_jobs, int) and n_jobs <= 1:
            n_jobs = False
        if not n_jobs:
            n_jobs = 1
        elif isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
            n_jobs = cpu_count() or 1

        # The ported estimator reads the samples (one per line) from the
        # standard input. Starting a process (e.g. the JVM) takes much longer
        # than a prediction, so each process predicts a batch of samples:
        n_batches = max(1, min(n_jobs, len(x) // MIN_BATCH_SIZE))
        batch_size = -(-len(x) // n_batches)  # rounded up
        batches = [
            '\n'.join(' '.join(map(str, e)) for e in x[i : i + batch_size])
            + '\n'
            for i in range(0, len(x), batch_size)
        ]

        def call(batch: str) -> list:
            return _system_call(cmd, batch)

        if len(batches) == 1:
            y = call(batches[0])
        else:
            # Threads are sufficient, because they only wait for the
            # subprocesses which execute the ported estimator:
            with ThreadPoolExecutor(len(batches)) as executor:
                y = [e for ys in executor.map(call, batches) for e in ys]
        y = list(zip(*y, strict=False))
        y_pred = np.array(y[0])
        meta_info = self._estimator.meta_info
        if 'labels' in meta_info and not meta_info.get('is_multilabel'):
            # The ported classifiers return the labels of the classes, so
            # convert them (e.g. `1` of JSON to `1.0`) to the type of the
            # labels of scikit-learn:
            classes = self._estimator.estimator.classes_
            if not isinstance(classes, list):
                classes = [classes]
            y_pred = y_pred.astype(np.result_type(*classes))
        y_proba: np.ndarray | list[np.ndarray] | None = None
        if len(y) > 1:
            if self._estimator.meta_info.get('is_multi_output'):
                # Multi-output classifiers return the probabilities of each
                # output (with different numbers of classes), so create one
                # array per output like scikit-learn:
                n_outputs = len(y[1][0])
                y_proba = [
                    np.array([p[i] for p in y[1]]) for i in range(n_outputs)
                ]
            else:
                y_proba = np.array(y[1])

        # Delete generated files finally:
        if delete_created_files:
            for path in created_files:
                if path and path.exists():
                    remove(str(path))
            if temp_dir:
                rmtree(temp_dir, ignore_errors=True)

        if len(y_pred) == 1:  # a single sample
            if isinstance(y_proba, list):  # multiple outputs
                return y_pred[0], [p[0] for p in y_proba]
            return y_pred[0], None if y_proba is None else y_proba[0]
        return y_pred, y_proba

    @staticmethod
    def _check_dependencies(language: enum.Language):
        for app in language.value.DEPENDENCIES:
            if not which(app):
                msg = f'Required dependency `{app}` is missing.'
                raise RuntimeError(msg)

    @staticmethod
    def _compile(
        src_path: Path,
        class_paths: list,
        created_files: list,
        language: enum.Language,
        template: enum.Template,
    ):
        """
        Execute a compilation.

        Parameters
        ----------
        src_path : Path
            The absolute path to the source files.
        class_paths : list
            A list of necessary class paths.
        created_files : list
            A linked list to save the paths of created files.
        language : Language
            The requested programming language.
        template
            The requested template.
        """
        if not language.value.CMD_COMPILE:
            return

        cmd_args = {}

        if language in (enum.Language.C, enum.Language.GO):
            dest_path = _get_binary_path(src_path)
            cmd_args['src_path'] = str(src_path)
            cmd_args['dest_path'] = str(dest_path)
            created_files.append(dest_path)

        elif language is enum.Language.CS:
            args, paths = language.value.prepare_compilation(src_path)
            cmd_args.update(args)
            created_files.extend(paths)

        elif language is enum.Language.JAVA:
            cmd_args['src_path'] = str(src_path)
            cmd_args['dest_dir'] = str(src_path.parent)
            class_paths.append(str(src_path.parent))
            created_files.append(src_path.parent / (src_path.stem + '.class'))

            # Dependencies:
            if template is enum.Template.EXPORTED:
                is_test = (
                    'SKLEARN_PORTER_PYTEST' in environ
                    and 'SKLEARN_PORTER_PYTEST_GSON_PATH' in environ
                )
                if is_test:
                    class_paths.append(
                        environ.get('SKLEARN_PORTER_PYTEST_GSON_PATH')
                    )
                else:
                    path = src_path.parent / 'gson.jar'
                    if not path.exists():
                        url = language.value.GSON_DOWNLOAD_URI
                        urllib.request.urlretrieve(url, str(path))
                        created_files.append(path)
                    class_paths.append(str(path))

            cmd_args['class_path'] = pathsep.join(class_paths)

        cmd = [arg.format(**cmd_args) for arg in language.value.CMD_COMPILE]
        L.info(f'Compilation command: `{join(cmd)}`')

        try:
            check_output(cmd, text=True, stderr=STDOUT)
        except OSError as e:  # e.g. a missing compiler
            msg = f'Command "{join(cmd)}" failed: {e}'
            raise exception.CompilationFailed(msg) from None
        except CalledProcessError as e:
            msg = (
                f'Command "{join(cmd)}" return with error '
                f'(code {e.returncode}):\n\n{e.output}'
            )
            if language is enum.Language.JAVA:
                if 'code too large' in e.output:
                    raise exception.CodeTooLarge(msg) from None
                elif 'too many constants' in e.output:
                    raise exception.TooManyConstants(msg) from None
            raise exception.CompilationFailed(msg) from None

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
        if isinstance(self._estimator.estimator, RegressorMixin):
            # Small deviations because of the floating point arithmetic:
            is_close = np.isclose(y_true, y_pred)
            if is_close.ndim > 1:  # multiple outputs
                is_close = is_close.all(axis=1)
            score = is_close.mean() if normalize else is_close.sum()
            return float(score)
        if np.ndim(y_true) > 1:  # multiple outputs or labels
            is_equal = (y_true == np.reshape(y_pred, np.shape(y_true))).all(
                axis=1
            )
            return float(is_equal.mean() if normalize else is_equal.sum())
        return float(accuracy_score(y_true, y_pred, normalize=normalize))

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

    def __repr__(self):
        """
        Get the status and basic information about
        the passed estimator and the local environment.

        Returns
        -------
        An overview of basic information.
        """
        report = f"""\
            estimator
            ---------
            name: {self._estimator.estimator_name}

            environment
            -----------
            platform       {platform}
            python         v{self.python_version}
            scikit-learn   v{sklearn_version}
            sklearn-porter v{self.porter_version}\
        """
        return dedent(report)


def _get_qualname(obj: object):
    return obj.__class__.__module__ + '.' + obj.__class__.__qualname__


def _is_windows() -> bool:
    return platform == 'win32'


def _get_binary_path(src_path: Path) -> Path:
    """Get the path of the compiled binary of a C or Go source file."""
    suffix = '.exe' if _is_windows() else ''
    return src_path.parent / (src_path.stem + suffix)


def _system_call(cmd: list[str], samples: str, n_retries: int = 3) -> list:
    """
    Separate helper function for parallel system calls.

    The command is executed directly without a shell, so the arguments
    (e.g. paths with spaces or special characters) don't need any quoting
    and the calls work on all platforms.

    Parameters
    ----------
    cmd : list of str
        The program and its arguments.
    samples : str
        The features of the samples, one sample per line, which are passed
        to the standard input.
    n_retries : int (default: 3)
        The number of retries after a failed call.

    Returns
    -------
    The predictions (and probabilities) of each sample.

    Raises
    ------
    ExecutionFailed
        If the call failed or didn't return valid JSON of each sample on
        every attempt.
    """
    n_samples = samples.count('\n')
    proc_stdout = proc_stderr = ''
    for attempt in range(n_retries + 1):
        if attempt:
            sleep(0.1 * 2**attempt)
        try:
            proc = run(
                cmd, input=samples, capture_output=True, encoding='utf-8'
            )
        except OSError as e:  # e.g. a missing or busy executable
            error = f'{type(e).__name__}: {e}'
            continue
        proc_stdout, proc_stderr = proc.stdout, proc.stderr
        if proc.returncode != 0:
            error = f'Exit code {proc.returncode}'
            continue
        try:
            outs = [loads(line) for line in proc_stdout.splitlines() if line]
        except JSONDecodeError as e:
            error = f'Invalid JSON ({e})'
            continue
        if len(outs) != n_samples:
            error = f'{len(outs)} results of {n_samples} samples'
            continue
        results = []
        for out in outs:
            result = [out.get('predict')]
            if 'predict_proba' in out.keys():
                result.append(out.get('predict_proba'))
            results.append(result)
        return results
    msg = (
        f'Command `{join(cmd)}` failed after {n_retries + 1} attempts.\n'
        f'{error}\n\nstdout:\n{proc_stdout}\nstderr:\n{proc_stderr}'
    )
    raise exception.ExecutionFailed(msg)


def show(language: str | enum.Language | None = None):
    """
    Show the supported estimators, programming languages
    and templates in a compact table.

    Parameters
    ----------
    language : Language
        The requested programming language.

    Returns
    -------
    Show the table.
    """
    languages = enum.LANGUAGES
    if language:
        language = enum.Language.convert(language)
        languages = {language.value.KEY: language.value}
    headers = ['Estimator'] + [lang.LABEL for lang in languages.values()]
    clazzes = Estimator.estimators()
    clazzes = {_get_qualname(c()): c() for c in clazzes}
    clazzes = dict(sorted(clazzes.items()))
    table = []
    templates = dict(attached='ᴀ', combined='ᴄ', exported='ᴇ')
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
            td = ' | '.join(td)
            tr.append(td)
        table.append(tr)
    return tabulate(
        table,
        headers=headers,
        tablefmt='presto',
        disable_numparse=True,
        colalign=['left'] + ['center'] * len(languages),
    )


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

    if isinstance(estimator, BaseEstimator):
        name = estimator.__class__.__name__
    elif isinstance(estimator, type):
        name = estimator.__name__
    else:
        return False

    cands = Estimator.estimators()
    cands = [c.__name__ for c in cands]

    if name not in cands:
        return False

    if language or template or method:
        pckg = f'sklearn_porter.estimator.{name}'
        clazz = getattr(import_module(pckg), name)
        support = clazz.SUPPORT
    else:
        return True

    if language:
        language = enum.Language.convert(language)
        if language in support.keys():
            if not template:
                return True
            else:
                template = enum.Template.convert(template)
                if template in support[language].keys():
                    if not method:
                        return True
                    else:
                        method = enum.Method.convert(method)
                        if method in support[language][template]:
                            return True
    return False


def port(
    estimator: BaseEstimator,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    to_json: bool = False,
    ignore_preprocessing: bool = False,
) -> str | tuple[str, str]:
    return Estimator(
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
    return Estimator(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
        ignore_preprocessing=ignore_preprocessing,
    ).save(directory=directory, to_json=to_json)


def make(
    estimator: BaseEstimator,
    x: list | np.ndarray,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    n_jobs: bool | int = True,
    directory: str | Path | None = None,
    delete_created_files: bool = True,
    check_dependencies: bool = True,
    ignore_preprocessing: bool = False,
) -> (
    tuple[np.int64, np.ndarray | None]
    | tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]
):
    return Estimator(
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
    return Estimator(
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
