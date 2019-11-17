# -*- coding: utf-8 -*-

import urllib.request
from abc import ABCMeta
from json import loads
from multiprocessing import Pool, cpu_count
from os import environ, remove
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from sys import platform as system_platform
from sys import stdout, version_info
from tempfile import mktemp
from textwrap import dedent
from time import sleep
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score

# sklearn-porter
from sklearn_porter import __version__
from sklearn_porter import enums as enum
from sklearn_porter import decorators as decorator
from sklearn_porter import exceptions as exception
from sklearn_porter.utils import Options, get_qualname


@decorator.aliased
class Estimator:
    """
    Main class which validates the passed estimator and
    coordinates the kind of estimator to a concrete subclass.
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        language: Optional[Union[str, enum.Language]] = None,
        template: Optional[Union[str, enum.Template]] = None,
        class_name: Optional[str] = None,
        converter: Optional[Callable[[object], str]] = None,
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
        # Log basic environment information:
        L.remove()
        logging_level = Options.get_option('logging.level')
        L.add(stdout, level=logging_level)

        python_version = '.'.join(map(str, version_info[:3]))
        L.debug('Platform: {}'.format(system_platform))
        L.debug('Python: v{}'.format(python_version))
        L.debug('Package: scikit-learn: v{}'.format(sklearn_version))
        L.debug('Package: sklearn-porter: v{}'.format(__version__))

        self.estimator = estimator  # see @estimator.setter

        def either_or(a, b):
            return a if a else b

        # Set defaults:
        self.language = either_or(language, self._estimator.DEFAULT_LANGUAGE)
        self.template = either_or(template, self._estimator.DEFAULT_TEMPLATE)
        self.class_name = either_or(class_name, self._estimator.estimator_name)
        self.converter = either_or(converter, lambda x: str(x))

    @property
    def estimator(self):
        return self._estimator.estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        orig_est = estimator
        orig_est = self._check_support(orig_est)
        if orig_est:
            self._estimator = self._load(orig_est)
        else:
            msg = 'The passed estimator is unknown'
            raise exception.NotSupportedYetError(msg)

    @property
    def language(self) -> str:
        return self._language.value.KEY

    @language.setter
    def language(self, language: Union[str, enum.Language]):
        language = enum.Language.convert(language)
        if self.can(self.estimator, language):
            self._language = language

    @property
    def template(self) -> str:
        return self._template.value

    @template.setter
    def template(self, name: Union[str, enum.Template]):
        template = enum.Template.convert(name)
        if self.can(self.estimator, self.language, template):
            self._template = template

    def _handle_default_kwargs(self, kwargs: Dict) -> Dict:
        keys = kwargs.keys()
        if 'language' in keys:
            self.language = kwargs.get('language')
        if 'template' in keys:
            self.template = kwargs.get('template')
        if 'class_name' in keys:
            class_name = kwargs.get('class_name')
            if isinstance(class_name, str):
                self.class_name = kwargs.get('class_name')
        if 'converter' in keys:
            converter = kwargs.get('converter')
            if isinstance(converter, Callable):
                self.converter = converter
        return kwargs

    @staticmethod
    def _check_support(estimator: BaseEstimator) -> Optional[BaseEstimator]:
        """
        Validate the estimator.

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
        # L.setLevel(Options.get_option('logging.level'))
        # L.level = Options.get_option('logging.level')

        est = estimator  # shorter <3
        qualname = get_qualname(est)

        L.debug(
            'Start validation of the passed '
            'estimator: `{}`.'.format(qualname)
        )

        # Check BaseEstimator:
        if not isinstance(est, BaseEstimator):
            msg = (
                'The passed estimator `{}` is not a '
                'valid base estimator of scikit-learn v{} .'
                ''.format(qualname, sklearn_version)
            )
            L.error(msg)
            raise ValueError(msg)

        # Check BaseEnsemble:
        if isinstance(est, BaseEnsemble):
            try:
                est.estimators_  # for sklearn > 0.19
            except AttributeError:
                raise exception.NotFittedEstimatorError(qualname)
            try:
                est.estimators_[0]  # for sklearn <= 0.18
            except IndexError:
                raise exception.NotFittedEstimatorError(qualname)

        # Check GridSearchCV and RandomizedSearchCV:
        L.debug('Check whether the estimator is embedded in an optimizer.')
        try:
            from sklearn.model_selection._search import (
                BaseSearchCV,
            )  # pylint: disable=protected-access
        except ImportError:
            msg = (
                'Your installed version of scikit-learn v{} '
                'does not support optimizers in general.'
                ''.format(sklearn_version)
            )
            L.warning(msg)
        else:
            if isinstance(est, BaseSearchCV):
                L.info('Yes, the estimator is embedded in an optimizer.')
                try:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.model_selection import RandomizedSearchCV
                except ImportError:
                    msg = (
                        'Your installed version of scikit-learn '
                        'v{} does not support `GridSearchCV` or '
                        '`RandomizedSearchCV`.'.format(sklearn_version)
                    )
                    L.warning(msg)
                else:
                    optimizers = (GridSearchCV, RandomizedSearchCV)
                    if isinstance(est, optimizers):
                        # pylint: disable=protected-access
                        is_fitted = (
                            hasattr(est, 'best_estimator_')
                            and est.best_estimator_
                        )
                        if is_fitted:
                            est = est.best_estimator_
                            est_qualname = get_qualname(est)
                            msg = (
                                'Extract the embedded estimator of '
                                'type `{}` from optimizer `{}`.'
                                ''.format(est_qualname, qualname)
                            )
                            L.info(msg)
                        # pylint: enable=protected-access
                        else:
                            msg = 'The embedded estimator is not fitted.'
                            L.error(msg)
                            raise ValueError(msg)
                    else:
                        msg = (
                            'The used optimizer `{}` is not supported '
                            'by sklearn-porter v{}. Try to extract the '
                            'internal estimator manually and pass it.'
                            ''.format(qualname, __version__)
                        )
                        L.error(msg)
                        raise ValueError(msg)
            else:
                L.info('No, the estimator is not embedded in an optimizer.')

        # Check Pipeline:
        L.debug('Check whether the estimator is embedded in a pipeline.')
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            msg = (
                'Your installed version of scikit-learn '
                'v{} does not support pipelines.'.format(sklearn_version)
            )
            L.warning(msg)
        else:
            if isinstance(est, Pipeline):
                L.info('Yes, the estimator is embedded in a pipeline.')
                # pylint: disable=protected-access
                has_est = (
                    hasattr(est, '_final_estimator') and est._final_estimator
                )
                if has_est:
                    est = est._final_estimator
                    est_qualname = get_qualname(est)
                    msg = (
                        'Extract the embedded estimator of type '
                        '`{}` from the pipeline.'.format(est_qualname)
                    )
                    L.info(msg)
                # pylint: enable=protected-access
                else:
                    msg = 'There is no final estimator is the pipeline.'
                    L.error(msg)
                    raise ValueError(msg)
            else:
                L.info('No, the estimator is not embedded in a pipeline.')

        # Check ClassifierMixin:
        L.debug('Check whether the estimator is a `ClassifierMixin`.')
        is_classifier = isinstance(est, ClassifierMixin)
        if is_classifier:
            L.debug('Yes, the estimator is type of `ClassifierMixin`.')
            return est
        L.debug('No, the estimator is not type of `ClassifierMixin`.')

        # Check RegressorMixin:
        L.debug('Check whether the estimator is a `RegressorMixin`.')
        is_regressor = isinstance(est, RegressorMixin)
        if is_regressor:
            L.debug('Yes, the estimator is type of `RegressorMixin`.')
            return est
        L.debug('No, the estimator is not type of `RegressorMixin`.')

        if not (is_classifier or is_regressor):
            msg = (
                'The passed estimator is neither '
                'a classifier nor a regressor.'
            )
            L.error(msg)
            raise ValueError(msg)

        return None

    @staticmethod
    def _load(estimator):
        """
        Load the right subclass to read the passed estimator.

        Parameters
        ----------
        estimator : Union[ClassifierMixin, RegressorMixin]
            Set a fitted base estimator of scikit-learn.

        Returns
        -------
        A subclass from `sklearn_porter.estimator.*` which
        represents and includes the original base estimator.
        """
        est = estimator  # shorter <3
        qualname = get_qualname(est)
        L.debug('Start loading the passed estimator: `{}`.'.format(qualname))

        name = est.__class__.__qualname__

        msg = (
            'Your installed version of scikit-learn v{} does not support '
            'the `{}` estimator. Please update your local installation '
            'of scikit-learn with `pip install -U scikit-learn`.'
        )

        # Classifiers:
        if name == 'DecisionTreeClassifier':
            from sklearn.tree.tree import (
                DecisionTreeClassifier as DecisionTreeClassifierClass,
            )

            if isinstance(est, DecisionTreeClassifierClass):
                from sklearn_porter.estimator.DecisionTreeClassifier import (
                    DecisionTreeClassifier,
                )

                return DecisionTreeClassifier(est)
        elif name == 'AdaBoostClassifier':
            from sklearn.ensemble.weight_boosting import (
                AdaBoostClassifier as AdaBoostClassifierClass,
            )

            if isinstance(estimator, AdaBoostClassifierClass):
                from sklearn_porter.estimator.AdaBoostClassifier import (
                    AdaBoostClassifier,
                )

                return AdaBoostClassifier(est)
        elif name == 'RandomForestClassifier':
            from sklearn.ensemble.forest import (
                RandomForestClassifier as RandomForestClassifierClass,
            )

            if isinstance(estimator, RandomForestClassifierClass):
                from sklearn_porter.estimator.RandomForestClassifier import (
                    RandomForestClassifier,
                )

                return RandomForestClassifier(est)
        elif name == 'ExtraTreesClassifier':
            from sklearn.ensemble.forest import (
                ExtraTreesClassifier as ExtraTreesClassifierClass,
            )

            if isinstance(estimator, ExtraTreesClassifierClass):
                from sklearn_porter.estimator.ExtraTreesClassifier import (
                    ExtraTreesClassifier,
                )

                return ExtraTreesClassifier(est)
        elif name == 'LinearSVC':
            from sklearn.svm.classes import LinearSVC as LinearSVCClass

            if isinstance(estimator, LinearSVCClass):
                from sklearn_porter.estimator.LinearSVC import LinearSVC

                return LinearSVC(est)
        elif name == 'SVC':
            from sklearn.svm.classes import SVC as SVCClass

            if isinstance(estimator, SVCClass):
                from sklearn_porter.estimator.SVC import SVC

                return SVC(est)
        elif name == 'NuSVC':
            from sklearn.svm.classes import NuSVC as NuSVCClass

            if isinstance(estimator, NuSVCClass):
                from sklearn_porter.estimator.NuSVC import NuSVC

                return NuSVC(est)
        elif name == 'KNeighborsClassifier':
            from sklearn.neighbors.classification import (
                KNeighborsClassifier as KNeighborsClassifierClass,
            )

            if isinstance(estimator, KNeighborsClassifierClass):
                from sklearn_porter.estimator.KNeighborsClassifier import (
                    KNeighborsClassifier,
                )

                return KNeighborsClassifier(est)
        elif name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB as GaussianNBClass

            if isinstance(estimator, GaussianNBClass):
                from sklearn_porter.estimator.GaussianNB import GaussianNB

                return GaussianNB(est)
        elif name == 'BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB as BernoulliNBClass

            if isinstance(estimator, BernoulliNBClass):
                from sklearn_porter.estimator.BernoulliNB import BernoulliNB

                return BernoulliNB(est)
        elif name == 'MLPClassifier':
            try:
                from sklearn.neural_network.multilayer_perceptron import (
                    MLPClassifier as MLPClassifierClass,
                )
            except ImportError:
                msg = msg.format(sklearn_version, name)
                L.error(msg)
                raise ValueError(msg)
            else:
                if isinstance(estimator, MLPClassifierClass):
                    from sklearn_porter.estimator.MLPClassifier import (
                        MLPClassifier,
                    )

                    return MLPClassifier(est)

        # Regressors:
        elif name == 'MLPRegressor':
            try:
                from sklearn.neural_network.multilayer_perceptron import (
                    MLPRegressor as MLPRegressorClass,
                )
            except ImportError:
                msg = msg.format(sklearn_version, name)
                L.error(msg)
                raise ValueError(msg)
            else:
                if isinstance(estimator, MLPRegressorClass):
                    from sklearn_porter.estimator.MLPRegressor import (
                        MLPRegressor,
                    )

                    return MLPRegressor(est)

        return None

    @decorator.alias('export')
    def port(self, to_json: bool = False, **kwargs) -> Union[str, Tuple[str]]:
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
        self._handle_default_kwargs(kwargs)
        return self._estimator.port(
            language=self._language,
            template=self._template,
            class_name=self.class_name,
            converter=self.converter,
            to_json=to_json
        )

    @decorator.alias('dump')
    def save(
        self,
        directory: Optional[Union[str, Path]] = None,
        to_json: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """
        Port a passed estimator to a target programming language and save them.

        Parameters
        ----------
        directory : Optional[Union[str, Path]] (default: current working dir)
            Set the directory where all generated files should be saved.
        to_json : bool (default: False)
            Return the result as JSON string.
        Returns
        -------
        The path(s) to the generated file(s).
        """
        self._handle_default_kwargs(kwargs)
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
        x: Union[List, np.ndarray],
        n_jobs: Optional[Union[bool, int]] = True,
        directory: Optional[Union[str, Path]] = None,
        final_deletion: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple[np.int64, np.ndarray], Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, None]]:
        """
        Make predictions with transpiled estimators locally.

        Parameters
        ----------
        x : Union[List, np.ndarray] of shape (n_samples, n_features) or (n_features)
            Input data.
        n_jobs : Union[bool, int] (default: True, which uses `count_cpus()`)
            The number of processes to make the predictions.
        directory : Optional[Union[str, Path]] (default: current working dir)
            Set the directory where all generated files should be saved.
        final_deletion : bool (default: False)
            Whether to delete the generated files finally or not.
        kwargs

        Returns
        -------
        Return the predictions and probabilities.
        """
        self._handle_default_kwargs(kwargs)
        if not directory:
            directory = mktemp()
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
            enum.Language.JS, enum.Language.PHP, enum.Language.RUBY
        ):
            cmd_args['src_path'] = str(src_path)

        cmd = cmd.format(**cmd_args)
        L.info('Execution command: `{}`'.format(cmd))

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

        if isinstance(n_jobs, int) and n_jobs <= 1:
            n_jobs = False
        if not n_jobs:
            y = list(map(_system_call, x))
        else:
            if isinstance(n_jobs, bool):
                n_jobs = cpu_count()
            if not isinstance(n_jobs, int):
                n_jobs = cpu_count()
            with Pool(n_jobs) as pool:
                y = pool.map(_system_call, x)
        y = list(zip(*y))
        y = list(map(np.array, y))

        # Delete generated files finally:
        if final_deletion:
            for path in created_files:
                if path and path.exists():
                    remove(str(path))

        if len(y) == 1:  # predict
            if len(y[0]) == 1:
                return y[0][0], None
            return y[0], None
        else:
            if len(y[0]) == 1:  # predict, predict_proba
                return y[0][0], y[1][0]
            return y[0], y[1]

    @staticmethod
    def _compile(
        src_path: Path,
        class_paths: List,
        created_files: List,
        language: enum.Language,
        template: Optional[enum.Template] = None,
    ):
        """
        Execute a compilation.

        Parameters
        ----------
        src_path : Path
            The absolute path to the source files.
        class_paths : List
            A list of necessary class paths.
        created_files : List
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
            created_files.append((src_path.parent / src_path.stem))

        elif language is enum.Language.JAVA:
            cmd_args['src_path'] = str(src_path)
            cmd_args['dest_dir'] = '-d {}'.format(str(src_path.parent))
            class_paths.append(str(src_path.parent))
            created_files.append((src_path.parent / (src_path.stem + '.class')))

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
        L.info('Compilation command: `{}`'.format(cmd))

        subp_args = dict(
            shell=True,
            universal_newlines=True,
            stderr=STDOUT,
            executable='/bin/bash'
        )
        try:
            check_output(cmd, **subp_args)
        except CalledProcessError as e:
            msg = 'Command "{}" return with error (code {}):\n\n{}'
            msg = msg.format(e.cmd, e.returncode, e.output)
            if language is enum.Language.JAVA and 'code too large' in e.output:
                raise exception.CodeTooLarge(msg)
            raise exception.CompilationFailed(msg)

    @decorator.alias('integrity_score')
    def test(
        self,
        x,
        n_jobs: Optional[Union[bool, int]] = True,
        directory: Optional[Union[str, Path]] = None,
        final_deletion: Optional[bool] = False,
        normalize: bool = True,
        **kwargs
    ):
        """
        Compute the accuracy of the ported classifier.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Input data.
        n_jobs : Union[bool, int] (default: True, which uses `count_cpus()`)
            The number of processes to make the predictions.
        directory : Optional[Union[str, Path]] (default: current working dir)
            Set the directory where all generated files should be saved.
        final_deletion : bool (default: False)
            Whether to delete the generated files finally or not.
        normalize : bool, default: True
            Whether to normalize the result or not.

        Returns
        -------
        score : Tuple[float, int]
            Return the relative and absolute number of correct
            classified samples.
        """
        self._handle_default_kwargs(kwargs)
        y_true = self._estimator.estimator.predict(x)
        y_pred = self.make(
            x,
            language=self._language,
            template=self._template,
            n_jobs=n_jobs,
            directory=directory,
            final_deletion=final_deletion,
        )
        y_pred = y_pred[0]  # only predicts
        return accuracy_score(y_true, y_pred, normalize=normalize)

    @staticmethod
    def classifiers() -> Tuple:
        """
        Get a set of supported and installed classifiers.

        Returns
        -------
        estimators : Tuple
            A set of supported classifiers.
        """

        # scikit-learn version < 0.18.0
        from sklearn.tree.tree import DecisionTreeClassifier
        from sklearn.ensemble.weight_boosting import AdaBoostClassifier
        from sklearn.ensemble.forest import RandomForestClassifier
        from sklearn.ensemble.forest import ExtraTreesClassifier
        from sklearn.svm.classes import LinearSVC
        from sklearn.svm.classes import SVC
        from sklearn.svm.classes import NuSVC
        from sklearn.neighbors.classification import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB

        classifiers = (
            AdaBoostClassifier,
            BernoulliNB,
            DecisionTreeClassifier,
            ExtraTreesClassifier,
            GaussianNB,
            KNeighborsClassifier,
            LinearSVC,
            NuSVC,
            RandomForestClassifier,
            SVC,
        )

        # scikit-learn version >= 0.18.0
        try:
            from sklearn.neural_network.multilayer_perceptron import (
                MLPClassifier,
            )
        except ImportError:
            pass
        else:
            classifiers += (MLPClassifier, )

        return classifiers

    @staticmethod
    def regressors() -> Tuple:
        """
        Get a set of supported and installed regressors.

        Returns
        -------
        estimators : Tuple
            A set of supported regressors.
        """

        # scikit-learn version < 0.18.0
        regressors = ()

        # scikit-learn version >= 0.18.0
        try:
            from sklearn.neural_network.multilayer_perceptron import (
                MLPRegressor,
            )
        except ImportError:
            pass
        else:
            regressors += (MLPRegressor, )

        return regressors

    @staticmethod
    def can(
        estimator: Union[BaseEstimator, ABCMeta],
        language: Optional[Union[str, enum.Language]] = None,
        template: Optional[Union[str, enum.Template]] = None,
        method: Optional[Union[str, enum.Method]] = None
    ) -> bool:
        """
        Check the support of the given arguments.

        Parameters
        ----------
        estimator : BaseEstimator or abstract ABCMeta class.
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
        elif isinstance(estimator, ABCMeta):
            name = estimator.__name__
        else:
            return False

        cands = Estimator.classifiers() + Estimator.regressors()
        cands = [c.__name__ for c in cands]

        if name not in cands:
            return False

        if language or template or method:
            pckg = 'sklearn_porter.estimator.{}'.format(name)
            module = __import__(pckg, globals(), locals(), [name], 0)
            clazz = getattr(module, name)
            support = getattr(clazz, 'SUPPORT')
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

    def __repr__(self):
        """
        Get the status and basic information about
        the passed estimator and the local environment.

        Returns
        -------
        An overview of basic information.
        """
        python_version = '.'.join(map(str, version_info[:3]))
        report = '''\
            estimator
            ---------
            name: {}
        
            environment
            -----------
            platform       {}
            python         v{}
            scikit-learn   v{}
            sklearn-porter v{}\
        '''.format(
            self._estimator.estimator_name,
            system_platform,
            python_version,
            sklearn_version,
            __version__,
        )
        return dedent(report)


def _system_call(cmd: str):
    """
    Separate helper function for multi-processed system calls.

    Parameters
    ----------
    cmd : str
        The command for the system call.

    Returns
    -------
    The output of subprocess.check_output.
    """
    subp_args = dict(
        shell=True,
        universal_newlines=True,
        stderr=STDOUT,
        executable='/bin/bash'
    )
    for idx in range(3):  # Try three times, b/c of `file is busy` issues.
        try:
            out = check_output(cmd, **subp_args)
        except CalledProcessError as e:
            sleep(0.01)
        else:
            out = str(out).strip()
            out = loads(out, encoding='utf-8')
            if 'predict_proba' in out.keys():
                return [out.get('predict'), out.get('predict_proba')]
            return [out.get('predict')]
