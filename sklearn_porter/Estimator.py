# -*- coding: utf-8 -*-
from pathlib import Path
from sys import version_info, platform as system_platform
from typing import Callable, Optional, Tuple, Union, List, Dict
from textwrap import dedent

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

# sklearn-porter
from sklearn_porter import __version__ as sklearn_porter_version
from sklearn_porter.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.utils import get_logger, get_qualname

L = get_logger(__name__)


class Estimator(EstimatorApiABC):
    """
    Main class which validates the passed estimator and
    coordinates the kind of estimator to a concrete subclass.
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            class_name: Optional[str] = None,
            method_name: Optional[str] = None,
            converter: Optional[Callable[[object], str]] = lambda x: str(x)
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
        method_name : str
            Change the default method name which will be used in the generated
            output. By default the name of the method/approach will be used,
            e.g. `predict`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        """
        # Log basic environment information:
        env_info = 'Environment: platform: {}; python: v{}; ' \
                   'scikit-learn: v{}; sklearn-porter: v{}'
        python_version = '.'.join(map(str, version_info[:3]))
        env_info = env_info.format(system_platform, python_version,
                                   sklearn_version, sklearn_porter_version)
        L.debug(env_info)

        # Set and load estimator:
        self._estimator = None
        self.estimator = estimator  # see @estimator.setter

        # Set default class name:
        if class_name:
            self.class_name = class_name
        else:
            self.class_name = self._estimator.estimator_name

        # Set default method name:
        if method_name:
            self.method_name = method_name
        else:
            self.method_name = None

        # Set the default converter:
        self.converter = converter

    @property
    def estimator(self):
        return self._estimator.estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        estimator = self._validate(estimator)
        if estimator:  # if valid
            self._estimator = self._load(estimator)

    @staticmethod
    def _validate(estimator: BaseEstimator):
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
        est = estimator  # shorter <3
        qualname = get_qualname(est)

        L.debug('Start validation of the passed estimator: `%s`.', qualname)

        # Check BaseEstimator:
        if not isinstance(est, BaseEstimator):
            msg = 'The passed estimator `{}` is not a ' \
                  'valid base estimator of scikit-learn v{} .'
            msg = msg.format(qualname, sklearn_version)
            L.error(msg)
            raise ValueError(msg)

        # Check GridSearchCV and RandomizedSearchCV:
        L.debug('Check whether the estimator is embedded in an optimizer.')
        try:
            from sklearn.model_selection._search \
                import BaseSearchCV  # pylint: disable=protected-access
        except ImportError:
            L.warn('Your installed version of scikit-learn '
                   'v% does not support optimizers in general.',
                   sklearn_version)
        else:
            if isinstance(est, BaseSearchCV):
                L.info('Yes, the estimator is embedded in an optimizer.')
                try:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.model_selection import RandomizedSearchCV
                except ImportError:
                    L.warn('Your installed version of scikit-learn '
                           'v% does not support `GridSearchCV` or '
                           '`RandomizedSearchCV`.', sklearn_version)
                else:
                    optimizers = (GridSearchCV, RandomizedSearchCV)
                    if isinstance(est, optimizers):
                        # pylint: disable=protected-access
                        is_fitted = hasattr(est, 'best_estimator_') and \
                                    est.best_estimator_
                        if is_fitted:
                            est = est.best_estimator_
                            est_qualname = get_qualname(est)
                            L.info('Extract the embedded estimator of '
                                   'type `%s` from optimizer `%s`.',
                                   est_qualname, qualname)
                        # pylint: enable=protected-access
                        else:
                            msg = 'The embedded estimator is not fitted.'
                            L.error(msg)
                            raise ValueError(msg)
                    else:
                        msg = 'The used optimizer `{}` is not supported ' \
                              'by sklearn-porter v{}. Try to extract the ' \
                              'internal estimator manually and pass it.' \
                              ''.format(qualname, sklearn_porter_version)
                        L.error(msg)
                        raise ValueError(msg)
            else:
                L.info('No, the estimator is not embedded in an optimizer.')

        # Check Pipeline:
        L.debug('Check whether the estimator is embedded in a pipeline.')
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            L.warn('Your installed version of scikit-learn '
                   'v% does not support pipelines.', sklearn_version)
        else:
            if isinstance(est, Pipeline):
                L.info('Yes, the estimator is embedded in a pipeline.')
                # pylint: disable=protected-access
                has_est = hasattr(est, '_final_estimator') and \
                          est._final_estimator
                if has_est:
                    est = est._final_estimator
                    est_qualname = get_qualname(est)
                    L.info('Extract the embedded estimator of type '
                           '`%s` from the pipeline.', est_qualname)
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
        else:
            L.debug('No, the estimator is not type of `ClassifierMixin`.')

        # Check RegressorMixin:
        L.debug('Check whether the estimator is a `RegressorMixin`.')
        is_regressor = isinstance(est, RegressorMixin)
        if is_regressor:
            L.debug('Yes, the estimator is type of `RegressorMixin`.')
            return est
        else:
            L.debug('No, the estimator is not type of `RegressorMixin`.')

        if not (is_classifier or is_regressor):
            msg = 'The passed estimator is neither ' \
                  'a classifier nor a regressor.'
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
        L.debug('Start loading the passed estimator: `%s`.', qualname)

        name = est.__class__.__qualname__

        msg = 'Your installed version of scikit-learn v{} does not support ' \
              'the `{}` estimator. Please update your local installation ' \
              'of scikit-learn with `pip install -U scikit-learn`.'

        # Classifiers:
        if name is 'DecisionTreeClassifier':
            from sklearn.tree.tree import DecisionTreeClassifier \
                as DecisionTreeClassifierClass
            if isinstance(est, DecisionTreeClassifierClass):
                from sklearn_porter.estimator.DecisionTreeClassifier \
                    import DecisionTreeClassifier
                return DecisionTreeClassifier(est)
        elif name is 'AdaBoostClassifier':
            from sklearn.ensemble.weight_boosting import AdaBoostClassifier \
                as AdaBoostClassifierClass
            if isinstance(estimator, AdaBoostClassifierClass):
                from sklearn_porter.estimator.AdaBoostClassifier \
                    import AdaBoostClassifier
                return AdaBoostClassifier(est)
        elif name is 'RandomForestClassifier':
            from sklearn.ensemble.forest import RandomForestClassifier \
                as RandomForestClassifierClass
            if isinstance(estimator, RandomForestClassifierClass):
                from sklearn_porter.estimator.RandomForestClassifier \
                    import RandomForestClassifier
                return RandomForestClassifier(est)
        elif name is 'ExtraTreesClassifier':
            from sklearn.ensemble.forest import ExtraTreesClassifier \
                as ExtraTreesClassifierClass
            if isinstance(estimator, ExtraTreesClassifierClass):
                from sklearn_porter.estimator.ExtraTreesClassifier \
                    import ExtraTreesClassifier
                return ExtraTreesClassifier(est)
        elif name is 'LinearSVC':
            from sklearn.svm.classes import LinearSVC as LinearSVCClass
            if isinstance(estimator, LinearSVCClass):
                from sklearn_porter.estimator.LinearSVC import LinearSVC
                return LinearSVC(est)
        elif name is 'SVC':
            from sklearn.svm.classes import SVC as SVCClass
            if isinstance(estimator, SVCClass):
                from sklearn_porter.estimator.SVC import SVC
                return SVC(est)
        elif name is 'NuSVC':
            from sklearn.svm.classes import NuSVC as NuSVCClass
            if isinstance(estimator, NuSVCClass):
                from sklearn_porter.estimator.NuSVC import NuSVC
                return NuSVC(est)
        elif name is 'KNeighborsClassifier':
            from sklearn.neighbors.classification import KNeighborsClassifier \
                as KNeighborsClassifierClass
            if isinstance(estimator, KNeighborsClassifierClass):
                from sklearn_porter.estimator.KNeighborsClassifier \
                    import KNeighborsClassifier
                return KNeighborsClassifier(est)
        elif name is 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB as GaussianNBClass
            if isinstance(estimator, GaussianNBClass):
                from sklearn_porter.estimator.GaussianNB import GaussianNB
                return GaussianNB(est)
        elif name is 'BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB as BernoulliNBClass
            if isinstance(estimator, BernoulliNBClass):
                from sklearn_porter.estimator.BernoulliNB import BernoulliNB
                return BernoulliNB(est)
        elif name is 'MLPClassifier':
            try:
                from sklearn.neural_network.multilayer_perceptron \
                    import MLPClassifier as MLPClassifierClass
            except ImportError:
                msg = msg.format(sklearn_version, name)
                L.error(msg)
                raise ValueError(msg)
            else:
                if isinstance(estimator, MLPClassifierClass):
                    from sklearn_porter.estimator.MLPClassifier \
                        import MLPClassifier
                    return MLPClassifier(est)

        # Regressors:
        elif name is 'MLPRegressor':
            try:
                from sklearn.neural_network.multilayer_perceptron \
                    import MLPRegressor as MLPRegressorClass
            except ImportError:
                msg = msg.format(sklearn_version, name)
                L.error(msg)
                raise ValueError(msg)
            else:
                if isinstance(estimator, MLPRegressorClass):
                    from sklearn_porter.estimator.MLPRegressor import \
                        MLPRegressor
                    return MLPRegressor(est)

        return None

    def port(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            **kwargs
    ) -> str:
        """
        Port or transpile a passed estimator to a target programming language.

        Parameters
        ----------
        method : str (default: 'predict')
            Set the target method.
        language : str (default: 'java')
            Set the target programming language.
        template : str (default: 'embedding')
            Set the kind of desired template.

        Returns
        -------
        The transpiled estimator in the target programming language.
        """
        locs = locals()
        locs.pop('self')
        locs.pop('kwargs')

        # Set defaults:
        kwargs = self._set_kwargs_defaults(kwargs, method_name=method)
        return self._estimator.port(**locs, **kwargs)

    def export(
            self,
            method: str = 'predict',
            language: str = 'java',
            template: str = 'combined',
            directory: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Union[str, List[str]]:
        """
        Port a passed estimator to a target programming language and save them.

        Parameters
        ----------
        method : str (default: 'predict')
            Set the target method.
        language : str (default: 'java')
            Set the target programming language.
        template : str (default: 'embedding')
            Set the kind of desired template.
        directory : Optional[Union[str, Path]] (default: current working dir)
            Set the directory where all generated files should be saved.

        Returns
        -------
        The path(s) to the generated file(s).
        """
        locs = locals()
        locs.pop('self')
        locs.pop('kwargs')

        # Set defaults:
        kwargs = self._set_kwargs_defaults(kwargs, method_name=method)
        return self._estimator.export(**locs, **kwargs)

    def _set_kwargs_defaults(self, kwargs: Dict, method_name: str) -> Dict:
        """
        Set default value for the methods `port` and `exports`.

        Parameters
        ----------
        kwargs : Dict
            The passed optional arguments.
        method_name : str
            The desired kind of method.

        Returns
        -------
        A dictionary with default values.
        """
        if self.method_name:
            method_name = self.method_name
        kwargs.setdefault('method_name', method_name)
        kwargs.setdefault('class_name', self.class_name)
        kwargs.setdefault('converter', self.converter)
        return kwargs

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
            from sklearn.neural_network.multilayer_perceptron \
                import MLPClassifier
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
            from sklearn.neural_network.multilayer_perceptron \
                import MLPRegressor
        except ImportError:
            pass
        else:
            regressors += (MLPRegressor, )

        return regressors

    def __repr__(self):
        python_version = '.'.join(map(str, version_info[:3]))
        report = '''\
            environment
            -----------
            platform       {}
            python         v{}
            scikit-learn   v{}
            sklearn-porter v{}
            
            estimator
            ---------
            name: {}\
        '''.format(system_platform, python_version, sklearn_version,
                   sklearn_porter_version, self._estimator.estimator_name)
        return dedent(report)
