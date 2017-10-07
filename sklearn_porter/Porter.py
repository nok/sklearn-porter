# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import sys
import subprocess as subp

import numpy as np

from sklearn.metrics import accuracy_score
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


class Porter(object):

    # Version:
    local_dir = os.path.dirname(__file__)
    version_file = os.path.join(local_dir, '__version__.txt')
    version = open(version_file).readlines().pop()
    if isinstance(version, bytes):
        version = version.decode('utf-8')
    __version__ = str(version).strip()

    def __init__(self, estimator, language='java', method='predict', **kwargs):
        # pylint: disable=unused-argument
        """
        Port a trained model to the syntax
        of a chosen programming language.

        Parameters
        ----------
        language : {'c', 'go', 'java', 'js', 'php', 'ruby'}, default 'java'
            The required target programming language.

        method : {'predict', 'predict_proba'}, default 'predict'
            The target prediction method.
        """

        # Check language support:
        language = str(language).strip().lower()
        if language not in ['c', 'go', 'java', 'js', 'php', 'ruby']:
            error = "The given language '{}' isn't supported.".format(language)
            raise AttributeError(error)
        self.target_language = language

        # Check method support:
        method = str(method).strip().lower()
        if method not in ['predict', 'predict_proba']:
            error = "The given method '{}' isn't supported.".format(method)
            raise AttributeError(error)
        self.target_method = method

        # Determine the local version of sklearn:
        from sklearn import __version__ as sklearn_ver
        sklearn_ver = str(sklearn_ver).split('.')
        sklearn_ver = [int(v) for v in sklearn_ver]
        major, minor = sklearn_ver[0], sklearn_ver[1]
        patch = sklearn_ver[2] if len(sklearn_ver) >= 3 else 0
        self.sklearn_ver = (major, minor, patch)

        # Extract estimator from 'Pipeline':
        # sklearn version >= 0.15.0
        if not hasattr(self, 'estimator') and self.sklearn_ver[:2] >= (0, 15):
            from sklearn.pipeline import Pipeline
            if isinstance(estimator, Pipeline):
                if hasattr(estimator, '_final_estimator') and \
                                estimator._final_estimator is not None:
                    self.estimator = estimator._final_estimator

        # Extract estimator from optimizer (GridSearchCV, RandomizedSearchCV):
        # sklearn version >= 0.19.0
        if not hasattr(self, 'estimator') and self.sklearn_ver[:2] >= (0, 19):
            from sklearn.model_selection._search import GridSearchCV
            from sklearn.model_selection._search import RandomizedSearchCV
            optimizers = (GridSearchCV, RandomizedSearchCV)
            if isinstance(estimator, optimizers):
                if hasattr(estimator, 'best_estimator_') and \
                        hasattr(estimator.best_estimator_, '_final_estimator'):
                    self.estimator = estimator.best_estimator_._final_estimator

        if not hasattr(self, 'estimator'):
            self.estimator = estimator

        # Determine the local supported estimators:
        self.supported_classifiers = self._classifiers
        self.supported_regressors = self._regressors

        # Read algorithm name and type:
        self.estimator_name = str(type(self.estimator).__name__)
        if isinstance(self.estimator, self.supported_classifiers):
            self.estimator_type = 'classifier'
        elif isinstance(self.estimator, self.supported_regressors):
            self.estimator_type = 'regressor'
        else:
            error = "Currently the given estimator '{estimator}' isn't" \
                    " supported.".format(**self.__dict__)
            raise ValueError(error)

        # Import estimator class:
        if sys.version_info[:2] < (3, 3):
            pckg = 'estimator.{estimator_type}.{estimator_name}'
            level = -1
        else:
            pckg = 'sklearn_porter.estimator.{estimator_type}.{estimator_name}'
            level = 0
        pckg = pckg.format(**self.__dict__)
        try:
            clazz = __import__(pckg, globals(), locals(),
                               [self.estimator_name], level)
            clazz = getattr(clazz, self.estimator_name)
        except ImportError:
            error = "Currently the given model '{algorithm_name}' " \
                    "isn't supported.".format(**self.__dict__)
            raise AttributeError(error)

        # Set target programming language:
        pwd = os.path.dirname(__file__)
        template_dir = os.path.join(pwd, 'estimator', self.estimator_type,
                                    self.estimator_name, 'templates',
                                    self.target_language)
        has_template = os.path.isdir(template_dir)
        if not has_template:
            error = "Currently there is no support of the combination " \
                    "of the estimator '{}' and the target programming " \
                    "language '{}'.".format(self.estimator_name,
                                            self.target_language)
            raise AttributeError(error)

        # Set target prediction method:
        has_method = self.target_method in \
                     set(getattr(clazz, 'SUPPORTED_METHODS'))
        if not has_method:
            error = "Currently the given model method" \
                    " '{}' isn't supported.".format(self.target_method)
            raise AttributeError(error)

        # Create instance with all parameters:
        self.template = clazz(**self.__dict__)

    def export(self, class_name='Brain', method_name='predict',
               use_repr=True, details=False, **kwargs):
        # pylint: disable=unused-argument
        """
        Transpile a trained model to the syntax of a
        chosen programming language.

        Parameters
        ----------
        :param class_name : string, default: 'Brain'
            The name for the ported class.

        :param method_name : string, default: 'predict'
            The name for the ported method.

        :param use_repr : bool, default: True
            Whether to use repr() for floating-point values or not.

        :param details : bool, default False
            Return additional data for the compilation
            and execution.

        Returns
        -------
        model : {mix}
            The ported model as string or a dictionary
            with further information.
        """
        output = self.template.export(class_name=class_name,
                                      method_name=method_name,
                                      use_repr=use_repr)
        if not details:
            return output

        language = self.target_language
        filename = Porter._get_filename(class_name, language)
        comp_cmd, exec_cmd = Porter._get_commands(filename,
                                                  class_name,
                                                  language)
        output = {
            'model': str(output),
            'filename': filename,
            'class_name': class_name,
            'method_name': method_name,
            'cmd': {
                'compilation': comp_cmd,
                'execution': exec_cmd
            },
            'algorithm': {
                'type': self.estimator_type,
                'name': self.estimator_name
            }
        }
        return output

    def port(self, class_name='Brain', method_name='predict', details=False):
        # pylint: disable=unused-argument
        """
        Transpile a trained model to the syntax of a
        chosen programming language.

        Parameters
        ----------
        :param class_name : string, default 'Brain'
            The name for the ported class.

        :param method_name : string, default 'predict'
            The name for the ported method.

        :param details : bool, default False
            Return additional data for the compilation
            and execution.

        Returns
        -------
        model : {mix}
            The ported model as string or a dictionary
            with further information.
        """
        loc = locals()
        loc.pop(str('self'))
        return self.export(**loc)

    @property
    def _classifiers(self):
        """
        Get a set of supported classifiers.

        Returns
        -------
        classifiers : {set}
            The set of supported classifiers.
        """

        # sklearn version < 0.18.0
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

        # sklearn version >= 0.18.0
        if self.sklearn_ver[:2] >= (0, 18):
            from sklearn.neural_network.multilayer_perceptron \
                import MLPClassifier
            classifiers += (MLPClassifier, )

        return classifiers

    @property
    def _regressors(self):
        """
        Get a set of supported regressors.

        Returns
        -------
        regressors : {set}
            The set of supported regressors.
        """

        # sklearn version < 0.18.0
        regressors = ()

        # sklearn version >= 0.18.0
        if self.sklearn_ver[:2] >= (0, 18):
            from sklearn.neural_network.multilayer_perceptron \
                import MLPRegressor
            regressors += (MLPRegressor, )

        return regressors

    def predict(self, X, class_name='Brain', method_name='predict',
                tnp_dir='tmp', keep_tmp_dir=False, use_repr=True):
        """
        Predict using the transpiled model.

        Parameters
        ----------
        :param X : {array-like}, shape (n_features) or (n_samples, n_features)
            The input data.

        :param class_name : string, default 'Brain'
            The name for the ported class.

        :param method_name : string, default 'predict'
            The name for the ported method.

        :param tnp_dir : string, default 'tmp'
            The path to the temporary directory for
            storing the transpiled (and compiled) model.

        :param keep_tmp_dir : bool, default False
            Whether to delete the temporary directory
            or not.
            
        :param use_repr : bool, default: True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
            y : int or array-like, shape (n_samples,)
            The predicted class or classes.
        """

        # Dependencies:
        if not hasattr(self, '_tested_dependencies'):
            self._test_dependencies()
            self._tested_dependencies = True

        # Support:
        if 'predict' not in set(self.template.SUPPORTED_METHODS):
            error = "Currently the given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)

        # Cleanup:
        subp.call(['rm', '-rf', tnp_dir])
        subp.call(['mkdir', tnp_dir])

        # Transpiled model:
        details = self.export(class_name=class_name,
                              method_name=method_name,
                              use_repr=use_repr, details=True)
        filename = Porter._get_filename(class_name, self.target_language)
        target_file = os.path.join(tnp_dir, filename)
        with open(target_file, str('w')) as file_:
            file_.write(details.get('model'))

        # Compilation command:
        comp_cmd = details.get('cmd').get('compilation')
        if comp_cmd is not None:
            comp_cmd = str(comp_cmd).split()
            subp.call(comp_cmd, cwd=tnp_dir)

        # Execution command:
        exec_cmd = details.get('cmd').get('execution')
        exec_cmd = str(exec_cmd).split()

        pred_y = None

        # Single feature set:
        if exec_cmd is not None and len(X.shape) == 1:
            full_exec_cmd = exec_cmd + [str(sample).strip() for sample in X]
            pred_y = subp.check_output(full_exec_cmd, stderr=subp.STDOUT,
                                       cwd=tnp_dir)
            pred_y = int(pred_y)

        # Multiple feature sets:
        if exec_cmd is not None and len(X.shape) > 1:
            pred_y = np.empty(X.shape[0], dtype=int)
            for idx, x in enumerate(X):
                full_exec_cmd = exec_cmd + [str(feature).strip() for feature in x]
                pred = subp.check_output(full_exec_cmd, stderr=subp.STDOUT,
                                         cwd=tnp_dir)
                pred_y[idx] = int(pred)

        # Cleanup:
        if not keep_tmp_dir:
            subp.call(['rm', '-rf', tnp_dir])

        return pred_y

    def predict_test(self, X, normalize=True, use_repr=True):
        """
        Compute the accuracy of the ported classifier.

        Parameters
        ----------
        :param X : ndarray, shape (n_samples, n_features)
            Input data.

        :param normalize : bool, optional (default=True)
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
            
        :param use_repr : bool, default: True
            Whether to use repr() for floating-point values or not.

        Returns
        -------
        score : float
            If ``normalize == True``, return the correctly classified samples
            (float), else it returns the number of correctly classified samples
            (int).
            The best performance is 1 with ``normalize == True`` and the number
            of samples with ``normalize == False``.
        """
        X = np.array(X)
        if not X.ndim > 1:
            X = np.array([X])
        y_true = self.estimator.predict(X)
        y_pred = self.predict(X, use_repr=use_repr)
        return accuracy_score(y_true, y_pred, normalize=normalize)

    def _test_dependencies(self):
        """
        Check all target programming and operating
        system dependencies.

        Parameters
        ----------
        :param language : {'c', 'go', 'java', 'js', 'php', 'ruby'}
            The target programming language.
        """
        lang = self.target_language

        if sys.platform in ('cygwin', 'win32', 'win64'):
            error = "The required dependencies aren't available on Windows."
            raise EnvironmentError(error)

        # Dependencies:
        depends = {
            'c': ('gcc'),
            'java': ('java', 'javac'),
            'js': ('node'),
            'go': ('go'),
            'php': ('php'),
            'ruby': ('ruby')
        }
        all_depends = depends.get(lang) + ('mkdir', 'rm')

        cmd = 'if hash {} 2/dev/null; then echo 1; else echo 0; fi'
        for exe in all_depends:
            cmd = cmd.format(exe)
            status = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            if sys.version_info >= (3, 3) and isinstance(status, bytes):
                status = status.decode('utf-8')
            status = str(status).strip()
            if status != '1':
                error = "The required application '{0}'" \
                        " isn't available.".format(exe)
                raise SystemError(error)

    @staticmethod
    def _get_filename(class_name, language):
        """
        Generate the specific filename.

        Parameters
        ----------
        :param class_name : str
            The used class name.

        :param language : {'c', 'go', 'java', 'js', 'php', 'ruby'}
            The target programming language.

        Returns
        -------
            filename : str
            The generated filename.
        """
        name = str(class_name).lower()
        lang = str(language)

        # Name:
        if language == 'java':
            name = name.capitalize()

        # Suffix:
        suffix = {
            'c': 'c', 'java': 'java', 'js': 'js',
            'go': 'go', 'php': 'php', 'ruby': 'rb'
        }
        suffix = suffix.get(lang, lang)

        # Filename:
        return '{}.{}'.format(name, suffix)

    @staticmethod
    def _get_commands(filename, class_name, language):
        """
        Generate the related compilation and
        execution commands.

        Parameters
        ----------
        :param filename : str
            The used filename.

        :param class_name : str
            The used class name.

        :param language : {'c', 'go', 'java', 'js', 'php', 'ruby'}
            The target programming language.

        Returns
        -------
            comp_cmd, exec_cmd : (str, str)
            The compilation and execution command.
        """
        cname = str(class_name).lower()
        fname = str(filename)
        lang = str(language)

        # Compilation variants:
        comp_vars = {
            # gcc brain.c -o brain
            'c': 'gcc {} -lm -o {}'.format(fname, cname),
            # javac Brain.java
            'java': 'javac {}'.format(fname)
        }
        comp_cmd = comp_vars.get(lang, None)

        # Execution variants:
        exec_vars = {
            # ./brain
            'c': os.path.join('.', cname),
            # java -classpath . Brain
            'java': 'java -classpath . {}'.format(cname.capitalize()),
            # node brain.js
            'js': 'node {}'.format(fname),
            # php -f brain.php
            'php': 'php -f {}'.format(fname),
            # ruby brain.rb
            'ruby': 'ruby {}'.format(fname)
        }
        exec_cmd = exec_vars.get(lang, None)

        return comp_cmd, exec_cmd
