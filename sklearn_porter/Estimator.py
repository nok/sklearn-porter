import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import import_module
from json import JSONDecodeError, loads
from os import cpu_count, environ, remove
from pathlib import Path
from shutil import rmtree, which
from subprocess import STDOUT, CalledProcessError, check_output, run
from sys import platform, stdout, version_info
from tempfile import mkdtemp
from textwrap import dedent
from time import sleep

# Additional installed modules:
import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaseEnsemble,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tabulate import tabulate

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter import meta
from sklearn_porter.utils import options


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
        """
        L.remove()  # remove basic logger
        L.add(stdout, level=options['logging.level'])

        self.python_version = '.'.join(map(str, version_info[:3]))
        self.porter_version = str(meta.__version__)

        L.debug(f'Platform: {platform}')
        L.debug(f'Python: v{self.python_version}')
        L.debug(f'Package: scikit-learn: v{sklearn_version}')
        L.debug(f'Package: sklearn-porter: v{self.porter_version}')

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
        orig_est = self._extract_est(estimator)
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
    def _extract_est(estimator: BaseEstimator) -> BaseEstimator | None:
        """
        Extract the original estimator.

        Check if the estimator is a valid base estimator of scikit-learn.
        Check if the estimator is embedded in an optimizer or pipeline.
        Check if the estimator is a classifier or regressor.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.

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
            estimators = getattr(est, 'estimators_', None)
            if not estimators:
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

        if not (is_classifier or is_regressor):
            msg = (
                f'The passed object of type `{qualname}` is neither '
                'a classifier nor a regressor.'
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
        estimator : ClassifierMixin or RegressorMixin
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
        clazzes = Estimator.classifiers() + Estimator.regressors()
        clazzes = {clazz.__name__: clazz for clazz in clazzes}
        if name in clazzes and isinstance(est, clazzes[name]):
            module = import_module(f'sklearn_porter.estimator.{name}')
            return getattr(module, name)(est)

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
        return self._estimator.port(
            language=self._language,
            template=self._template,
            class_name=self.class_name,
            converter=self.converter,
            to_json=to_json,
        )

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
        shell_executable: str = '/bin/bash',
        **kwargs,
    ) -> (
        tuple[np.int64, np.ndarray]
        | tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, None]
    ):
        """
        Make predictions with transpiled estimators locally.

        Parameters
        ----------
        x : list or np.ndarray of shape (n_samples, n_features) or (n_features)
            Input data.
        n_jobs : bool or int (default: True, which uses `os.cpu_count()`)
            The number of threads to make the predictions.
        directory : str or Path, optional (default: a temporary directory)
            Set the directory where all generated files should be saved.
        delete_created_files : bool (default: True)
            Whether to delete the generated files finally or not.
        check_dependencies : bool (default: True)
            Check whether all required applications are in the PATH or not.
        shell_executable : str (default: '/bin/bash')
            The shell which should be used for the Popen system calls.
        kwargs

        Returns
        -------
        Return the predictions and probabilities.
        """
        self._check_kwargs(kwargs)
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
        cmd = language.value.CMD_EXECUTE
        cmd_args = {}

        if language in (enum.Language.C, enum.Language.GO):
            cmd_args['dest_path'] = str(src_path.parent / src_path.stem)
        elif language is enum.Language.JAVA:
            if bool(class_paths):
                cmd_args['class_path'] = '-cp ' + ':'.join(class_paths)
            cmd_args['dest_path'] = str(src_path.stem)
        elif language in (
            enum.Language.JS,
            enum.Language.PHP,
            enum.Language.RUBY,
        ):
            cmd_args['src_path'] = str(src_path)

        paths_to_check = [
            cmd_args[key]
            for key in ('src_path', 'dest_path')
            if key in cmd_args
        ]
        for path_to_check in paths_to_check:
            for _ in range(10):
                try:
                    fp = open(path_to_check)
                except OSError:
                    sleep(0.1)
                else:
                    fp.close()
                    break

        cmd = cmd.format(**cmd_args)
        L.info(f'Execution command: `{cmd}`')

        # Model data:
        json_path = ' ' if not data_path else ' ' + str(data_path) + ' '

        # Features:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        x = x.tolist()

        # Command:
        x = [cmd + json_path + ' '.join(list(map(str, e))) for e in x]
        calls = partial(_system_call, executable=shell_executable)

        if isinstance(n_jobs, int) and n_jobs <= 1:
            n_jobs = False

        if not n_jobs:
            y = list(map(calls, x))
        else:
            if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
                n_jobs = cpu_count() or 1
            # Threads are sufficient, because they only wait for the
            # subprocesses which execute the ported estimator:
            with ThreadPoolExecutor(n_jobs) as executor:
                y = list(executor.map(calls, x))
        y = list(zip(*y, strict=False))
        y = list(map(np.array, y))

        # Delete generated files finally:
        if delete_created_files:
            for path in created_files:
                if path and path.exists():
                    remove(str(path))
            if temp_dir:
                rmtree(temp_dir, ignore_errors=True)

        if len(y) == 1:  # predict
            if len(y[0]) == 1:
                return y[0][0], None
            return y[0], None
        else:
            if len(y[0]) == 1:  # predict, predict_proba
                return y[0][0], y[1][0]
            return y[0], y[1]

    @staticmethod
    def _check_dependencies(language: enum.Language):
        # is_windows = platform in ('cygwin', 'win32', 'win64')
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
        cmd = language.value.CMD_COMPILE

        if not cmd:
            return

        cmd_args = {}

        if language in (enum.Language.C, enum.Language.GO):
            cmd_args['src_path'] = str(src_path)
            cmd_args['dest_path'] = str(src_path.parent / src_path.stem)
            created_files.append(src_path.parent / src_path.stem)

        elif language is enum.Language.JAVA:
            cmd_args['src_path'] = str(src_path)
            cmd_args['dest_dir'] = f'-d {src_path.parent}'
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

            if bool(class_paths):
                cmd_args['class_path'] = '-cp ' + ':'.join(class_paths)

        cmd = cmd.format(**cmd_args)
        L.info(f'Compilation command: `{cmd}`')

        try:
            check_output(
                cmd,
                shell=True,
                text=True,
                stderr=STDOUT,
                executable='/bin/bash',
            )
        except CalledProcessError as e:
            msg = (
                f'Command "{e.cmd}" return with error '
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
            The number of threads to make the predictions.
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
            DecisionTreeClassifier,
            ExtraTreesClassifier,
            GaussianNB,
            KNeighborsClassifier,
            LinearSVC,
            LogisticRegression,
            MLPClassifier,
            NuSVC,
            Perceptron,
            RandomForestClassifier,
            RidgeClassifier,
            SGDClassifier,
            SVC,
        )

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
            DecisionTreeRegressor,
            ElasticNet,
            ExtraTreesRegressor,
            Lasso,
            LinearRegression,
            MLPRegressor,
            RandomForestRegressor,
            Ridge,
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


def _system_call(cmd: str, executable='/bin/bash', n_retries: int = 3):
    """
    Separate helper function for parallel system calls.

    Parameters
    ----------
    cmd : str
        The command for the system call.
    executable : str (default: '/bin/bash')
        The shell which should be used for the system call.
    n_retries : int (default: 3)
        The number of retries after a failed call.

    Returns
    -------
    The predictions (and probabilities) of the executed estimator.

    Raises
    ------
    ExecutionFailed
        If the call failed or didn't return valid JSON on every attempt.
    """
    for attempt in range(n_retries + 1):
        if attempt:
            sleep(0.1 * 2**attempt)
        proc = run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            executable=executable,
        )
        if proc.returncode != 0:
            error = f'Exit code {proc.returncode}'
            continue
        try:
            out = loads(proc.stdout)
        except JSONDecodeError as e:
            error = f'Invalid JSON ({e})'
            continue
        result = [out.get('predict')]
        if 'predict_proba' in out.keys():
            result.append(out.get('predict_proba'))
        return result
    msg = (
        f'Command `{cmd}` failed after {n_retries + 1} attempts.\n'
        f'{error}\n\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
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
    clazzes = Estimator.classifiers() + Estimator.regressors()
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

    cands = Estimator.classifiers() + Estimator.regressors()
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
) -> str | tuple[str, str]:
    return Estimator(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
    ).port(to_json)


def save(
    estimator: BaseEstimator,
    language: str | enum.Language | None = None,
    template: str | enum.Template | None = None,
    class_name: str | None = None,
    converter: Callable[[object], str] | None = None,
    directory: str | Path | None = None,
    to_json: bool = False,
) -> str | tuple[str, str]:
    return Estimator(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
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
    shell_executable: str = '/bin/bash',
) -> (
    tuple[np.int64, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, None]
):
    return Estimator(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
    ).make(
        x,
        n_jobs=n_jobs,
        directory=directory,
        delete_created_files=delete_created_files,
        check_dependencies=check_dependencies,
        shell_executable=shell_executable,
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
) -> float:
    return Estimator(
        estimator,
        language=language,
        template=template,
        class_name=class_name,
        converter=converter,
    ).test(
        x,
        n_jobs=n_jobs,
        directory=directory,
        delete_created_files=delete_created_files,
        normalize=normalize,
    )
