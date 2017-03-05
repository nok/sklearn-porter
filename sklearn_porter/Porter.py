# -*- coding: utf-8 -*-

import os
import sys
import subprocess as subp

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


class Porter:
    __version__ = '0.4.0'

    def __init__(self, model, language='java', method='predict', **kwargs):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param language : string (default='java')
            The required target programming language.
            Set: ['c', 'go', 'java', 'js', 'php', 'ruby']

        :param method : string (default='predict')
            The name and type of the prediction method.
            Set: ['predict']
        """
        self.model = model
        self.target_language = language
        self.target_method = method

        shared_properties = ('algo_name', 'algo_type')
        for prop in shared_properties:
            key = prop.lower()
            if hasattr(self, key):
                setattr(self, key, getattr(self, key))

        # Import model class:
        try:
            package = '{algo_type}.{algo_name}'.format(**self.__dict__)
            level = -1 if sys.version_info < (3, 3) else 1
            clazz = __import__(package, globals(), locals(),
                               [self.algo_name], level)
            clazz = getattr(clazz, self.algo_name)
        except ImportError:
            error = "The given model '{algo_name}' isn't" \
                    " supported.".format(**self.__dict__)
            raise AttributeError(error)

        algorithm = clazz(**self.__dict__)

        # Target programming language:
        language = str(language).strip().lower()
        pwd = os.path.dirname(__file__)
        template_dir = os.path.join(pwd, self.algo_type, self.algo_name,
                                    'templates', language)
        has_template = os.path.isdir(template_dir)
        if not has_template:
            error = "The given target programming language" \
                    " '{}' isn't supported.".format(language)
            raise AttributeError(error)
        self.target_language = language

        # Prediction method:
        has_method = method in set(algorithm.SUPPORTED_METHODS)
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format(method)
            raise AttributeError(error)
        self.target_method = method
        algorithm.target_method = method

        self.algorithm = algorithm

    @property
    def algo_name(self):
        """Get the algorithm class name."""
        return str(type(self.model).__name__)

    @property
    def algo_type(self):
        """Get algorithm type, which is either a classifier or regressor."""
        if isinstance(self.model, self.classifiers):
            return 'classifier'
        # if isinstance(self.model, self.regressors):
        #     return 'regressor'
        error = "The given model '{model}' isn't" \
                " supported.".format(**self.__dict__)
        raise ValueError(error)

    @property
    def classifiers(self):
        """List of supported classifiers."""
        classifiers = (  # sklearn version < 0.18.0
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
        from sklearn import __version__ as version  # sklearn version >= 0.18.0
        version = version.split('.')
        version = [int(v) for v in version]
        major, minor = version[0], version[1]
        if major > 0 or (major == 0 and minor >= 18):
            from sklearn.neural_network.multilayer_perceptron \
                import MLPClassifier
            classifiers += (MLPClassifier, )
        return classifiers

    def export(self, class_name='Brain',
               method_name='predict',
               details=False, **kwargs):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param class_name : string (default='Brain')
            The name for the ported class.

        :param method_name : string (default='predict')
            The name for the ported method.

        :param details : bool (default=False)
            Return additional data for compiling and execution.

        Returns
        -------
        :return : string
            The ported model as string.
        """
        class_name = str(class_name).strip()
        method_name = str(method_name).strip()
        language = self.target_language

        print('LANG', language)


        transpiled_model = self.algorithm.export(class_name=class_name,
                                                 method_name=method_name)
        if not details:
            return transpiled_model

        filename = self._build_filename(class_name)
        comp_cmd, exec_cmd = self._build_commands(filename, class_name, language)
        output = {
            'model': transpiled_model,
            'filename': filename,
            'class_name': class_name,
            'method_name': method_name,
            'cmd': {
                'compilation': comp_cmd,
                'execution': exec_cmd
            },
            'algorithm': {
                'type': self.algo_type,
                'name': self.algo_name
            }
        }
        return output

    def port(self, class_name='Brain', method_name='predict', details=False):
        loc = locals()
        loc.pop('self')
        return self.export(**loc)

    def predict(self, X, class_name='Brain', method_name='predict',
                tnp_dir='tmp', keep_tmp_dir=False):
        has_method = 'predict' in set(self.algorithm.SUPPORTED_METHODS)
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)
        details = self.export(class_name=class_name, method_name=method_name,
                              details=True)
        subp.call(['rm', '-rf', tnp_dir])
        subp.call(['mkdir', tnp_dir])

        filename = self._build_filename(class_name)
        target_file = os.path.join(tnp_dir, filename)
        with open(target_file, 'w') as f:
            transpiled_model = details.get('model')
            f.write(transpiled_model)

        comp_cmd = details.get('cmd').get('compilation')
        if comp_cmd is not None:
            comp_cmd = str(comp_cmd).split()
            subp.call(comp_cmd, cwd=tnp_dir)

        prediction = -1  # default value
        exec_cmd = details.get('cmd').get('execution')
        exec_cmd = str(exec_cmd).split()
        if exec_cmd is not None:
            full_exec_cmd = exec_cmd + [str(f).strip() for f in X]
            prediction = subp.check_output(full_exec_cmd, stderr=subp.STDOUT,
                                           cwd=tnp_dir)
            prediction = int(prediction)

        if not keep_tmp_dir:
            subp.call(['rm', '-rf', tnp_dir])

        return prediction

    @staticmethod
    def _dependencies(language):
        dependencies = {
            'c': ['gcc'],
            'java': ['java', 'javac'],
            'js': ['node'],
            # 'go': [],
            'php': ['php'],
            'ruby': ['ruby']
        }
        dependencies = dependencies.get(language) + ['mkdir', 'rm']
        test_cmd = 'if hash {} 2/dev/null; then echo 1; else echo 0; fi'
        for exe in dependencies:
            cmd = test_cmd.format(exe)
            available = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                error = "The required application '{0}'" \
                        " isn't available.".format(exe)
                raise SystemError(error)
        return True

    def _build_filename(self, class_name):
        language = self.target_language

        # Name:
        name = class_name.lower()
        if language == 'java':
            name = name.capitalize()

        # Suffix:
        suffix = {
            'c': 'c', 'java': 'java', 'js': 'js',
            'go': 'go', 'php': 'php', 'ruby': 'rb'
        }
        suffix = suffix.get(language, language)

        # Filename:
        return '{}.{}'.format(name, suffix)

    def _build_commands(self, filename, class_name, language):
        # Get compiling command:
        compilation_variants = {
            # gcc brain.c -o brain
            'c': 'gcc {} -lm -o {}'.format(filename, class_name.lower()),
            # javac Brain.java
            'java': 'javac {}'.format(filename)
        }
        compilation_cmd = compilation_variants.get(language, None)

        # Build execution command:
        execution_variants = {
            # ./brain
            'c': os.path.join('.', class_name.lower()),
            # java -classpath . Brain
            'java': 'java -classpath . {}'.format(class_name.capitalize()),
            # node brain.js
            'js': 'node {}'.format(filename)
            # TODO: Add go exec command
        }
        execution_cmd = execution_variants.get(language, None)

        return compilation_cmd, execution_cmd
