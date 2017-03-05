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

        shared_properties = ('algorithm_name', 'algorithm_type')
        for prop in shared_properties:
            if hasattr(self, prop):
                setattr(self, prop, getattr(self, prop))

        # Import model class:
        package = '{algorithm_type}.' \
                  '{algorithm_name}'.format(**self.__dict__)
        level = -1 if sys.version_info < (3, 3) else 1
        try:
            clazz = __import__(package, globals(), locals(),
                               [self.algorithm_name], level)
            clazz = getattr(clazz, self.algorithm_name)
        except ImportError:
            error = "The given model '{algorithm_name}' " \
                    "isn't supported.".format(**self.__dict__)
            raise AttributeError(error)

        # Set target programming language:
        language = str(language).strip().lower()
        pwd = os.path.dirname(__file__)
        template_dir = os.path.join(pwd, self.algorithm_type,
                                    self.algorithm_name,
                                    'templates', language)
        has_template = os.path.isdir(template_dir)
        if not has_template:
            error = "The given target programming language" \
                    " '{}' isn't supported.".format(language)
            raise AttributeError(error)
        self.target_language = language

        # Set target prediction method:
        has_method = method in set(getattr(clazz, 'SUPPORTED_METHODS'))
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format(method)
            raise AttributeError(error)
        self.target_method = method

        # Create instance with all parameters:
        self.algorithm = clazz(**self.__dict__)

        self.tested_env_dependencies = False


    @property
    def algorithm_name(self):
        """Get the algorithm class name."""
        return str(type(self.model).__name__)

    @property
    def algorithm_type(self):
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
        model = self.algorithm.export(class_name=class_name,
                                      method_name=method_name)
        if not details:
            return model

        language = self.target_language
        filename = Porter.get_filename(class_name, language)
        comp_cmd, exec_cmd = Porter.get_commands(filename,
                                                 class_name,
                                                 language)
        output = {
            'model': model,
            'filename': filename,
            'class_name': class_name,
            'method_name': method_name,
            'cmd': {
                'compilation': comp_cmd,
                'execution': exec_cmd
            },
            'algorithm': {
                'type': self.algorithm_type,
                'name': self.algorithm_name
            }
        }
        return output

    def port(self, class_name='Brain', method_name='predict', details=False):
        loc = locals()
        loc.pop('self')
        return self.export(**loc)

    def predict(self, X, class_name='Brain', method_name='predict',
                tnp_dir='tmp', keep_tmp_dir=False):
        if not self.tested_env_dependencies:
            Porter.test_dependencies(self.target_language)

        has_method = 'predict' in set(self.algorithm.SUPPORTED_METHODS)
        if not has_method:
            error = "The given model method" \
                    " '{}' isn't supported.".format('predict')
            raise AttributeError(error)
        details = self.export(class_name=class_name,
                              method_name=method_name,
                              details=True)
        subp.call(['rm', '-rf', tnp_dir])
        subp.call(['mkdir', tnp_dir])

        filename = Porter.get_filename(class_name, self.target_language)
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
    def test_dependencies(language):
        lang = str(language)

        # Dependencies:
        depends = {
            'c': ['gcc'],
            'java': ['java', 'javac'],
            'js': ['node'],
            # 'go': [],
            'php': ['php'],
            'ruby': ['ruby']
        }
        all_depends = depends.get(lang) + ['mkdir', 'rm']

        # Test:
        cmd = 'if hash {} 2/dev/null; then echo 1; else echo 0; fi'
        for exe in all_depends:
            cmd = cmd.format(exe)
            available = subp.check_output(cmd, shell=True,
                                          stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                error = "The required application '{0}'" \
                        " isn't available.".format(exe)
                raise SystemError(error)
        return True

    @staticmethod
    def get_filename(class_name, language):
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
    def get_commands(filename, class_name, language):
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
