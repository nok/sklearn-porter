# -*- coding: utf-8 -*-

import os
import sys
import subprocess

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

    def __init__(self, model, language='java', method='predict', *args):
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
        self.algorithm = type(model).__name__
        self.algorithm_type = 'classifier'

        from sklearn import __version__ as sklearn_version
        version = sklearn_version.split('.')
        major, minor = int(version[0]), int(version[1])

        # scikit-learn version < 0.18.0
        methods = (
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
        if major > 0 or (major == 0 and minor >= 18):
            from sklearn.neural_network.multilayer_perceptron \
                import MLPClassifier
            methods += (MLPClassifier, )

        if not isinstance(model, methods):
            error = "The given model '{algorithm}' isn't" \
                    " supported.".format(**self.__dict__)
            raise ValueError(error)

        # Import model class:
        try:
            package = '{algorithm_type}.{algorithm}'.format(**self.__dict__)
            level = -1 if sys.version_info < (3, 3) else 1
            clazz = __import__(package, globals(), locals(),
                               [self.algorithm], level)
            clazz = getattr(clazz, self.algorithm)
        except ImportError:
            error = "The given model '{algorithm}' isn't" \
                    " supported.".format(**self.__dict__)
            raise AttributeError(error)
        instance = clazz(model, **self.__dict__)

        # Target programming language:
        language = str(language).strip().lower()
        pwd = os.path.dirname(__file__)
        template_dir = os.path.join(pwd, self.algorithm_type, self.algorithm,
                                    'templates', language)
        has_template = os.path.isdir(template_dir)
        if not has_template:
            error = "The given target programming language" \
                    " '{}' isn't supported.".format(language)
            raise AttributeError(error)
        self.target_language = language

        # Prediction method:
        has_method = method in set(instance.SUPPORTED_METHODS)
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format(method)
            raise AttributeError(error)
        self.target_method = method
        instance.target_method = method

        self.model = instance

    def export(self, class_name='Brain', method_name='predict', details=False, **kwargs):
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

        model = self.model.export(class_name=class_name, method_name=method_name)

        print('model', model)

        if not details:
            return str(model)

        filename = self._build_filename(self.class_name)
        compilation_cmd, execution_cmd = self._build_commands(filename)
        return str(model), compilation_cmd, execution_cmd

    def port(self, class_name='Brain', method_name='predict', details=False):
        return self.export(**locals())

    def predict(self, X, class_name='Brain', method_name='predict',
                tnp_dir='tmp', keep_tmp_dir=False):
        has_method = 'predict' in set(self.model.SUPPORTED_METHODS)
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)

        # options = locals()
        # options.update({'details': True})
        model, comp_cmd, exec_cmd = self.export(class_name=class_name,
                                                method_name=method_name,
                                                details=True)
        subprocess.call(['rm', tnp_dir])
        subprocess.call(['mkdir', tnp_dir])
        cd_cmd = ['cd', tnp_dir]

        filename = self._build_filename(class_name)
        target_file = os.path.join(tnp_dir, filename)
        with open(target_file, 'w') as f:
            f.write(model)

        if comp_cmd is not None:
            subprocess.call(cd_cmd + ['&&'] + comp_cmd.split(' '))
        if exec_cmd is not None:
            subprocess.call(cd_cmd)
            exec_cmd = [exec_cmd] + [str(f).strip() for f in X]
            pred = subprocess.check_output(exec_cmd, stderr=subprocess.STDOUT)
            pred = int(pred)

        if not keep_tmp_dir:
            subprocess.call(['rm', tnp_dir])

        return pred

    @staticmethod
    def _dependencies(self, language):
        language = str(language).lower().strip()
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
            available = subprocess.check_output(cmd,shell=True,
                                                stderr=subprocess.STDOUT)
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

    def _build_commands(self, filename):
        class_name = self.class_name
        language = self.target_language

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
