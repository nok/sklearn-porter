# -*- coding: utf-8 -*-

import os
import sys
import types

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell

# from sklearn_porter.language import *


class Porter(object):

    def __init__(self, estimator, language='java', method='predict', **kwargs):
        # pylint: disable=unused-argument
        """
        Transpile a trained estimator to the
        chosen target programming language.

        Parameters
        ----------
        language : {'c', 'go', 'java', 'js', 'php', 'ruby'}, default: 'java'
            The required target programming language.

        method : {'predict', 'predict_proba'}, default: 'predict'
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
            error = "Currently the chosen target programming language '{}' " \
                    "isn't supported for the estimator '{}'." \
                    "".format(self.estimator_name, self.target_language)
            raise AttributeError(error)

        # Set target prediction method:
        has_method = self.target_method in \
                     set(getattr(clazz, 'SUPPORTED_METHODS'))
        if not has_method:
            error = "Currently the chosen model method" \
                    " '{}' isn't supported.".format(self.target_method)
            raise AttributeError(error)

        self._tested_dependencies = False

        # Create instance with all parameters:
        self.template = clazz(**self.__dict__)

    def export(self, class_name=None, method_name=None,
               num_format=lambda x: str(x), details=False, **kwargs):
        # pylint: disable=unused-argument
        """
        Transpile a trained model to the syntax of a
        chosen programming language.

        Parameters
        ----------
        :param class_name : string, default: None
            The name for the ported class.

        :param method_name : string, default: None
            The name for the ported method.

        :param num_format : lambda x, default: lambda x: str(x)
            The representation of the floating-point values.

        :param details : bool, default False
            Return additional data for the compilation and execution.

        Returns
        -------
        model : {mix}
            The ported model as string or a dictionary
            with further information.
        """

        if class_name is None or class_name == '':
            class_name = self.estimator_name

        if method_name is None or method_name == '':
            method_name = self.target_method

        if isinstance(num_format, types.LambdaType):
            self.template._num_format = num_format

        output = self.template.export(class_name=class_name,
                                      method_name=method_name, **kwargs)
        if not details:
            return output

        language = self.target_language
        filename = Porter._get_filename(class_name, language)
        comp_cmd, exec_cmd = Porter._get_commands(filename, class_name,
                                                  language)
        output = {
            'estimator': str(output),
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

    def port(self, class_name=None, method_name=None,
             num_format=lambda x: str(x), details=False, **kwargs):
        # pylint: disable=unused-argument
        """
        Transpile a trained model to the syntax of a
        chosen programming language.

        Parameters
        ----------
        :param class_name : string, default: None
            The name for the ported class.

        :param method_name : string, default: None
            The name for the ported method.

        :param num_format : lambda x, default: lambda x: str(x)
            The representation of the floating-point values.

        :param details : bool, default: False
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
            from sklearn.neural_network \
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
            from sklearn.neural_network \
                import MLPRegressor
            regressors += (MLPRegressor, )

        return regressors

    def predict(self, X, class_name=None, method_name=None, tnp_dir='tmp',
                keep_tmp_dir=False, num_format=lambda x: str(x)):
        """
        Predict using the transpiled model.

        Parameters
        ----------
        :param X : {array-like}, shape (n_features) or (n_samples, n_features)
            The input data.

        :param class_name : string, default: None
            The name for the ported class.

        :param method_name : string, default: None
            The name for the ported method.

        :param tnp_dir : string, default: 'tmp'
            The path to the temporary directory for
            storing the transpiled (and compiled) model.

        :param keep_tmp_dir : bool, default: False
            Whether to delete the temporary directory
            or not.

        :param num_format : lambda x, default: lambda x: str(x)
            The representation of the floating-point values.

        Returns
        -------
            y : int or array-like, shape (n_samples,)
            The predicted class or classes.
        """

        if class_name is None:
            class_name = self.estimator_name

        if method_name is None:
            method_name = self.target_method

        # Dependencies:
        if not self._tested_dependencies:
            self._test_dependencies()
            self._tested_dependencies = True

        # Support:
        if 'predict' not in set(self.template.SUPPORTED_METHODS):
            error = "Currently the given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)

        # Cleanup:
        Shell.call('rm -rf {}'.format(tnp_dir))
        Shell.call('mkdir {}'.format(tnp_dir))

        # Transpiled model:
        details = self.export(class_name=class_name,
                              method_name=method_name,
                              num_format=num_format,
                              details=True)
        filename = Porter._get_filename(class_name, self.target_language)
        target_file = os.path.join(tnp_dir, filename)
        with open(target_file, str('w')) as file_:
            file_.write(details.get('estimator'))

        # Compilation command:
        comp_cmd = details.get('cmd').get('compilation')
        if comp_cmd is not None:
            Shell.call(comp_cmd, cwd=tnp_dir)

        # Execution command:
        exec_cmd = details.get('cmd').get('execution')
        exec_cmd = str(exec_cmd).split()

        pred_y = None

        # Single feature set:
        if exec_cmd is not None and len(X.shape) == 1:
            full_exec_cmd = exec_cmd + [str(sample).strip() for sample in X]
            pred_y = Shell.check_output(full_exec_cmd, cwd=tnp_dir)
            pred_y = int(pred_y)

        # Multiple feature sets:
        if exec_cmd is not None and len(X.shape) > 1:
            pred_y = np.empty(X.shape[0], dtype=int)
            for idx, features in enumerate(X):
                full_exec_cmd = exec_cmd + [str(f).strip() for f in features]
                pred = Shell.check_output(full_exec_cmd, cwd=tnp_dir)
                pred_y[idx] = int(pred)

        # Cleanup:
        if not keep_tmp_dir:
            Shell.call('rm -rf {}'.format(tnp_dir))

        return pred_y

    def integrity_score(self, X, method='predict', normalize=True,
                        num_format=lambda x: str(x)):
        """
        Compute the accuracy of the ported classifier.

        Parameters
        ----------
        :param X : ndarray, shape (n_samples, n_features)
            Input data.

        :param method : string, default: 'predict'
            The method which should be tested.

        :param normalize : bool, default: True
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

        :param num_format : lambda x, default: lambda x: str(x)
            The representation of the floating-point values.

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

        method = str(method).strip().lower()
        if method not in ['predict', 'predict_proba']:
            error = "The given method '{}' isn't supported.".format(method)
            raise AttributeError(error)

        if method == 'predict':
            y_true = self.estimator.predict(X)
            y_pred = self.predict(X, tnp_dir='tmp_integrity_score',
                                  keep_tmp_dir=True, num_format=num_format)
            return accuracy_score(y_true, y_pred, normalize=normalize)

        return False

    def _test_dependencies(self):
        """
        Check all target programming and operating
        system dependencies.
        """
        lang = self.target_language

        # Dependencies:
        deps = {
            'c': ['gcc'],
            'java': ['java', 'javac'],
            'js': ['node'],
            'go': ['go'],
            'php': ['php'],
            'ruby': ['ruby']
        }
        current_deps = deps.get(lang) + ['mkdir', 'rm']
        Environment.check_deps(current_deps)

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
        name = str(class_name).strip()
        lang = str(language)

        # Name:
        if language in ['java', 'php']:
            name = "".join([name[0].upper() + name[1:]])

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
        cname = str(class_name)
        fname = str(filename)
        lang = str(language)

        # Compilation variants:
        comp_vars = {
            # gcc brain.c -o brain
            'c': 'gcc {} -lm -o {}'.format(fname, cname),
            # javac Brain.java
            'java': 'javac {}'.format(fname),
            # go build -o brain brain.go
            'go': 'go build -o {} {}.go'.format(cname, cname)
        }
        comp_cmd = comp_vars.get(lang, None)

        # Execution variants:
        exec_vars = {
            # ./brain
            'c': os.path.join('.', cname),
            # java -classpath . Brain
            'java': 'java -classpath . {}'.format(cname),
            # node brain.js
            'js': 'node {}'.format(fname),
            # php -f Brain.php
            'php': 'php -f {}'.format(fname),
            # ruby brain.rb
            'ruby': 'ruby {}'.format(fname),
            # ./brain
            'go': os.path.join('.', cname),
        }
        exec_cmd = exec_vars.get(lang, None)

        return comp_cmd, exec_cmd
