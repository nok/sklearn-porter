import urllib.request
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from json import loads
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
from sklearn.base import BaseEstimator

# sklearn-porter
from sklearn_porter import decorators as decorator
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter import meta
from sklearn_porter.utils import densify, options, tabify

# The minimal number of samples which `make()` passes to one process of the
# ported estimator, because starting a process takes much longer than a
# prediction (or a transformation):
MIN_BATCH_SIZE = 100


@decorator.aliased
class Base:
    """
    Base class of `Estimator`, `Transformer` and `Pipeline`, which validates
    the passed estimator, coordinates it to a concrete subclass and ports,
    compiles and executes it.
    """

    # The kind of the passed estimator (e.g. in the overview of `__repr__`),
    # which is also the package of the ported classes, e.g. `estimator` for
    # `sklearn_porter.estimator.*`:
    KIND: str

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
        """
        self._init_environment()
        self.estimator = estimator  # see @estimator.setter

        # Set defaults:
        self.language = language or self._estimator.DEFAULT_LANGUAGE
        self.template = template or self._estimator.DEFAULT_TEMPLATE
        self.class_name = class_name or self._estimator.estimator_name
        self.converter = converter or str

    def _init_environment(self):
        """Set up the logger and read the versions of the environment."""
        L.remove()  # remove basic logger
        L.add(stdout, level=options['logging.level'])

        self.python_version = '.'.join(map(str, version_info[:3]))
        self.porter_version = str(meta.__version__)

        L.debug(f'Platform: {platform}')
        L.debug(f'Python: v{self.python_version}')
        L.debug(f'Package: scikit-learn: v{sklearn_version}')
        L.debug(f'Package: sklearn-porter: v{self.porter_version}')

    @property
    def _name(self) -> str:
        """The name of the ported estimator (e.g. in error messages)."""
        return self._estimator.estimator_name

    @property
    def estimator(self):
        return self._estimator.estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        orig_est = self._extract(estimator)
        self._estimator = self._load_est(orig_est)

    @property
    def language(self) -> str:
        return self._language.value.KEY

    @language.setter
    def language(self, language: str | enum.Language):
        language = enum.Language.convert(language)
        if self.supports(self.estimator, language):
            self._language = language
        else:
            name = self._name
            msg = (
                f'The passed language `{language.value.KEY}` is not '
                f'supported for the {self.KIND} `{name}`.'
            )
            raise exception.NotSupportedYetError(msg)

    @property
    def template(self) -> str:
        return self._template.value

    @template.setter
    def template(self, name: str | enum.Template):
        template = enum.Template.convert(name)
        if self.supports(self.estimator, self.language, template):
            self._template = template
        else:
            name = self._name
            language = enum.Language.convert(self.language)
            msg = (
                f'The passed template `{template.value}` is not '
                f'supported for the {self.KIND} `{name}` '
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

    def _extract(self, estimator: BaseEstimator) -> BaseEstimator | None:
        """
        Validate the passed estimator and extract the one to port.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.

        Returns
        -------
        The estimator to port.
        """
        raise NotImplementedError

    @staticmethod
    def _registry() -> tuple:
        """
        Get the supported classes of scikit-learn of this kind.

        Returns
        -------
        The supported classes, e.g. `Estimator.estimators()`.
        """
        raise NotImplementedError

    @classmethod
    def _load_est(cls, estimator):
        """
        Load the right subclass to read the passed estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.

        Returns
        -------
        A subclass from `sklearn_porter.estimator.*` (or
        `sklearn_porter.transformer.*`) which represents and includes the
        original base estimator.
        """
        return _load_ported(estimator, cls._registry(), cls.KIND)

    @classmethod
    def supports(
        cls,
        estimator: BaseEstimator | type,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        method: str | enum.Method | None = None,
    ) -> bool:
        """
        Check the support of the given arguments for this kind.

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

        if name not in {clazz.__name__ for clazz in cls._registry()}:
            return False

        if not (language or template or method):
            return True

        pckg = f'sklearn_porter.{cls.KIND}.{name}'
        support = getattr(import_module(pckg), name).SUPPORT

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

    def can(
        self,
        language: str | enum.Language | None = None,
        template: str | enum.Template | None = None,
        method: str | enum.Method | None = None,
    ) -> bool:
        return self.supports(
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

    def _run(
        self,
        x,
        n_jobs: bool | int,
        directory: str | Path | None,
        delete_created_files: bool,
        check_dependencies: bool,
        parse: Callable[[str], list],
        **kwargs,
    ) -> list:
        """
        Compile and execute the ported estimator with the passed samples.

        Parameters
        ----------
        x : list or np.ndarray of shape (n_samples, n_features) or (n_features)
            Input data.
        n_jobs : bool or int
            The maximal number of processes.
        directory : str or Path, optional (default: a temporary directory)
            Set the directory where all generated files should be saved.
        delete_created_files : bool
            Whether to delete the generated files finally or not.
        check_dependencies : bool
            Check whether all required applications are in the PATH or not.
        parse : Callable
            Parse the standard output of the ported estimator to one result
            per sample.
        kwargs

        Returns
        -------
        The results of all samples in the order of the samples.

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
                stacklevel=3,
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
            x = x.toarray()
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
        # than a prediction, so each process gets a batch of samples:
        n_batches = max(1, min(n_jobs, len(x) // MIN_BATCH_SIZE))
        batch_size = -(-len(x) // n_batches)  # rounded up
        batches = [
            '\n'.join(' '.join(map(str, e)) for e in x[i : i + batch_size])
            + '\n'
            for i in range(0, len(x), batch_size)
        ]

        def call(batch: str) -> list:
            return _system_call(cmd, batch, parse)

        if len(batches) == 1:
            y = call(batches[0])
        else:
            # Threads are sufficient, because they only wait for the
            # subprocesses which execute the ported estimator:
            with ThreadPoolExecutor(len(batches)) as executor:
                y = [e for ys in executor.map(call, batches) for e in ys]

        # Delete generated files finally:
        if delete_created_files:
            for path in created_files:
                if path and path.exists():
                    remove(str(path))
            if temp_dir:
                rmtree(temp_dir, ignore_errors=True)

        return y

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

    def __repr__(self):
        """
        Get the status and basic information about
        the passed estimator and the local environment.

        Returns
        -------
        An overview of basic information.
        """
        report = f"""\
            {self.KIND}
            {'-' * len(self.KIND)}
            name: {self._name}

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


def _load_ported(estimator: BaseEstimator, clazzes: tuple, package: str):
    """
    Load the subclass of a package to read the passed estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        Set a fitted base estimator of scikit-learn.
    clazzes : tuple
        The supported classes of scikit-learn.
    package : str
        The package of the subclasses: `estimator` or `transformer`.

    Returns
    -------
    A subclass from `sklearn_porter.<package>.*` which represents and
    includes the original base estimator.
    """
    est = estimator  # shorter <3
    qualname = _get_qualname(est)
    L.debug(f'Start loading the passed {package}: `{qualname}`.')

    name = est.__class__.__qualname__
    by_name = {clazz.__name__: clazz for clazz in clazzes}
    if name in by_name and isinstance(est, by_name[name]):
        module = import_module(f'sklearn_porter.{package}.{name}')
        # Read the model data from dense arrays, because the ported
        # estimators don't support sparse matrices:
        ported_est = getattr(module, name)(densify(est))
        # Keep the original estimator for the predictions of scikit-learn
        # (e.g. SVMs fitted on sparse data need sparse support vectors):
        ported_est.estimator = est
        return ported_est

    msg = f'The passed {package} `{name}` is not supported.'
    raise exception.NotSupportedYetError(msg)


def _parse_predictions(stdout: str) -> list:
    """
    Parse the predictions (and probabilities) of the samples.

    Parameters
    ----------
    stdout : str
        The standard output of a ported estimator: one JSON object per
        sample and line.

    Returns
    -------
    The predictions (and probabilities) of each sample.
    """
    results = []
    for out in (loads(line) for line in stdout.splitlines() if line):
        result = [out.get('predict')]
        if 'predict_proba' in out.keys():
            result.append(out.get('predict_proba'))
        results.append(result)
    return results


def _parse_values(stdout: str) -> list:
    """
    Parse the values of the samples (e.g. the transformed features).

    Parameters
    ----------
    stdout : str
        The standard output of a ported transformer: the values of one sample
        per line, separated by spaces.

    Returns
    -------
    The values of each sample.
    """
    return [
        [float(value) for value in line.split()]
        for line in stdout.splitlines()
        if line.strip()
    ]


def _system_call(
    cmd: list[str],
    samples: str,
    parse: Callable[[str], list] = _parse_predictions,
    n_retries: int = 3,
) -> list:
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
    parse : Callable (default: the parser of the predictions)
        Parse the standard output to one result per sample.
    n_retries : int (default: 3)
        The number of retries after a failed call.

    Returns
    -------
    The results (e.g. the predictions and probabilities) of each sample.

    Raises
    ------
    ExecutionFailed
        If the call failed or didn't return a valid result of each sample on
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
            results = parse(proc_stdout)
        except ValueError as e:  # e.g. invalid JSON or numbers
            error = f'Invalid output ({e})'
            continue
        if len(results) != n_samples:
            error = f'{len(results)} results of {n_samples} samples'
            continue
        return results
    msg = (
        f'Command `{join(cmd)}` failed after {n_retries + 1} attempts.\n'
        f'{error}\n\nstdout:\n{proc_stdout}\nstderr:\n{proc_stderr}'
    )
    raise exception.ExecutionFailed(msg)
