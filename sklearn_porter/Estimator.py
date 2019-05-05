# -*- coding: utf-8 -*-

from sys import version_info, platform as system_platform
from typing import Union, Tuple, Optional, Callable
from textwrap import dedent
from logging import Logger, ERROR

# scikit-learn
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

# sklearn-porter
from sklearn_porter import __version__ as sklearn_porter_version
from sklearn_porter.EstimatorInterApiABC import EstimatorInterApiABC
from sklearn_porter.utils import get_logger, get_qualname


class Estimator(EstimatorInterApiABC):
    """
    Main class which validates the passed estimator and
    coordinates the kind of estimator to a correct subclass.
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            logger: Union[Logger, int] = ERROR
    ):
        """
        Validate and coordinate the passed estimator for transpiling.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.
        logger : logging.Logger or logging level (default: logging.ERROR)
            Set a logger or logging level for logging.
        """
        self.logger = get_logger(__name__, logger)

        # Log basic environment information:
        env_info = 'Environment: platform: {}; python: v{}; ' \
                   'scikit-learn: v{}; sklearn-porter: v{}'
        python_version = '.'.join(map(str, version_info[:3]))
        env_info = env_info.format(system_platform, python_version,
                                   sklearn_version, sklearn_porter_version)
        self.logger.debug(env_info)

        # Set and load estimator:
        self._estimator = None
        self.estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator):
        estimator = self._validate(estimator, logger=self.logger)
        if estimator:
            self._estimator = self._load(estimator, logger=self.logger)

    @staticmethod
    def _validate(estimator: BaseEstimator, logger: Union[Logger, int] = ERROR):
        """
        Validate the estimator.

        Check if the estimator is a valid base estimator of scikit-learn.
        Check if the estimator is embedded in an optimizer or pipeline.
        Check if the estimator is a classifier or regressor.

        Parameters
        ----------
        estimator : BaseEstimator
            Set a fitted base estimator of scikit-learn.
        logger : logging.Logger or logging level (default: logging.ERROR)
            Set a logger or logging level for logging.

        Returns
        -------
        A valid base estimator or None.
        """
        logger = get_logger(__name__, logger)

        est = estimator  # shorter <3
        qualname = get_qualname(est)

        logger.debug('Start validation of the passed'
                     ' estimator: `%s`.', qualname)

        # Check BaseEstimator:
        if not isinstance(est, BaseEstimator):
            msg = 'The passed estimator `{}` is not a ' \
                  'valid base estimator of scikit-learn v{} .'
            msg = msg.format(qualname, sklearn_version)
            logger.error(msg)
            raise ValueError(msg)

        # Check GridSearchCV and RandomizedSearchCV:
        logger.debug('Check whether the estimator is embedded in an optimizer.')
        try:
            from sklearn.model_selection._search \
                import BaseSearchCV  # pylint: disable=protected-access
        except ImportError:
            logger.warn('Your installed version of scikit-learn '
                        'v% does not support optimizers in general.',
                        sklearn_version)
        else:
            if isinstance(est, BaseSearchCV):
                logger.info('Yes, the estimator is embedded in an optimizer.')
                try:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.model_selection import RandomizedSearchCV
                except ImportError:
                    logger.warn('Your installed version of scikit-learn '
                                'v% does not support `GridSearchCV` or'
                                '`RandomizedSearchCV`.', sklearn_version)
                else:
                    optimizers = (GridSearchCV, RandomizedSearchCV)
                    if isinstance(est, optimizers):
                        # pylint: disable=protected-access
                        is_fitted = hasattr(est, 'best_estimator_') and \
                                    hasattr(est.best_estimator_,
                                            '_final_estimator') and \
                                    est.best_estimator_._final_estimator
                        if is_fitted:
                            est = est.best_estimator_._final_estimator
                            est_qualname = get_qualname(est)
                            logger.info('Extract the embedded estimator of '
                                        'type `%s` from optimizer `%s`.',
                                        est_qualname, qualname)
                        # pylint: enable=protected-access
                        else:
                            msg = 'The embedded estimator is not fitted.'
                            logger.error(msg)
                            raise ValueError(msg)
                    else:
                        msg = 'The used optimizer `{}` is not supported ' \
                              'by sklearn-porter v{}. Try to extract the ' \
                              'internal estimator manually and pass it.' \
                              ''.format(qualname, sklearn_porter_version)
                        logger.error(msg)
                        raise ValueError(msg)
            else:
                logger.info('No, the estimator is not '
                            'embedded in an optimizer.')

        # Check Pipeline:
        logger.debug('Check whether the estimator is embedded in a pipeline.')
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.warn('Your installed version of scikit-learn '
                        'v% does not support pipelines.', sklearn_version)
        else:
            if isinstance(est, Pipeline):
                logger.info('Yes, the estimator is embedded in a pipeline.')
                # pylint: disable=protected-access
                is_fitted = hasattr(est, '_final_estimator') and \
                            est._final_estimator
                if is_fitted:
                    est = est._final_estimator
                    est_qualname = get_qualname(est)
                    logger.info('Extract the embedded estimator of type '
                                '`%s` from the pipeline.', est_qualname)
                # pylint: enable=protected-access
                else:
                    msg = 'The embedded estimator is not fitted.'
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                logger.info('No, the estimator is not embedded in a pipeline.')

        # Check ClassifierMixin:
        logger.debug('Check whether the estimator is a `ClassifierMixin`.')
        is_classifier = isinstance(est, ClassifierMixin)
        if is_classifier:
            logger.debug('Yes, the estimator is type of `ClassifierMixin`.')
            return est
        else:
            logger.debug('No, the estimator is not type of `ClassifierMixin`.')

        # Check RegressorMixin:
        logger.debug('Check whether the estimator is a `RegressorMixin`.')
        is_regressor = isinstance(est, RegressorMixin)
        if is_regressor:
            logger.debug('Yes, the estimator is type of `RegressorMixin`.')
            return est
        else:
            logger.debug('No, the estimator is not type of `RegressorMixin`.')

        if not (is_classifier or is_regressor):
            msg = 'The passed estimator is neither ' \
                  'a classifier nor a regressor.'
            logger.error(msg)
            raise ValueError(msg)

        return None

    @staticmethod
    def _load(estimator, logger: Union[Logger, int] = ERROR):
        logger = get_logger(__name__, logger)

        est = estimator  # shorter <3
        qualname = get_qualname(est)
        logger.debug('Start loading the passed estimator: `%s`.', qualname)

        name = est.__class__.__qualname__

        # Classifiers:
        if name is 'DecisionTreeClassifier':
            from sklearn.tree.tree import DecisionTreeClassifier \
                as DecisionTreeClassifierClass
            if isinstance(est, DecisionTreeClassifierClass):
                from sklearn_porter.estimator.DecisionTreeClassifier \
                    import DecisionTreeClassifier
                return DecisionTreeClassifier(est, logger=logger)
        elif name is 'AdaBoostClassifier':
            from sklearn.ensemble.weight_boosting import AdaBoostClassifier
            if isinstance(estimator, AdaBoostClassifier):
                pass
        elif name is 'RandomForestClassifier':
            from sklearn.ensemble.forest import RandomForestClassifier
            if isinstance(estimator, RandomForestClassifier):
                pass
        elif name is 'ExtraTreesClassifier':
            from sklearn.ensemble.forest import ExtraTreesClassifier
            if isinstance(estimator, ExtraTreesClassifier):
                pass
        elif name is 'LinearSVC':
            from sklearn.svm.classes import LinearSVC
            if isinstance(estimator, LinearSVC):
                pass
        elif name is 'SVC':
            from sklearn.svm.classes import SVC
            if isinstance(estimator, SVC):
                pass
        elif name is 'NuSVC':
            from sklearn.svm.classes import NuSVC
            if isinstance(estimator, NuSVC):
                pass
        elif name is 'KNeighborsClassifier':
            from sklearn.neighbors.classification import \
                KNeighborsClassifier
            if isinstance(estimator, KNeighborsClassifier):
                pass
        elif name is 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            if isinstance(estimator, GaussianNB):
                pass
        elif name is 'BernoulliNB':
            from sklearn.naive_bayes import BernoulliNB
            if isinstance(estimator, BernoulliNB):
                pass
        elif name is 'MLPClassifier':
            try:
                from sklearn.neural_network.multilayer_perceptron \
                    import MLPClassifier
            except ImportError:
                pass
            else:
                if isinstance(estimator, MLPClassifier):
                    pass

        # Regressors:
        elif name is 'MLPRegressor':
            try:
                from sklearn.neural_network.multilayer_perceptron \
                    import MLPRegressor
            except ImportError:
                msg = 'Your installed version of scikit-learn v{} does ' \
                      'not support the `MLPRegressor` estimator. Please ' \
                      'update your local installation with `pip install ' \
                      '-U scikit-learn`.'.format(sklearn_version)
                logger.error(msg)
                raise ValueError(msg)
            else:
                if isinstance(estimator, MLPRegressor):
                    from sklearn_porter.estimator.MLPRegressor import \
                        MLPRegressor
                    return MLPRegressor(est, logger=logger)

        return None

    def port(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        """
        Port or transpile a passed estimator to a target programming language.

        Parameters
        ----------
        method : str (default: 'predict')
            Set the target method.
        to : str (default: 'java')
            Set the target programming language.
        with_num_format : Callable[[object], str] (default: `lambda x: str(x)`)
            Set a function for custom numeric conversions.
        with_class_name : str
            Set a custom class name in the result.
        with_method_name : str
            Set a custom method name in the result.

        Returns
        -------
        The transpiled estimator in the target programming language.
        """
        locs = locals()
        locs.pop('self')
        return self._estimator.port(**locs)

    def export(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        locs = locals()
        locs.pop('self')
        return self.port(**locs)

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
                   sklearn_porter_version, self._estimator.default_class_name)
        return dedent(report)
