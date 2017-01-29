# -*- coding: utf-8 -*-

import os
import sys

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

    __version__ = '0.3.2'

    def __init__(self, language="java", method_name='predict', class_name='Tmp',
                 with_details=False):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param language : string (default='java')
            The required syntax ['c', 'go', 'java', 'js'].

        :param method_name : string (default='predict')
            The name of the prediction method.

        :param class_name : string (default='Tmp')
            The name of the environment class.

        :param with_details : bool (default=False)
            Return additional useful information or not.
        """
        self.language = language
        self.method_name = method_name
        self.class_name = class_name
        self.with_details = with_details

    def port(self, model):
        """
        Port a trained model to the syntax of a chosen programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).

        Returns
        -------
        :return : string
            The ported model as string.
        """
        cat, name = self.get_model_data(model)
        path = '.'.join([cat, name])  # e.g.: classifier.LinearSVC
        level = -1 if sys.version_info < (3, 3) else 1
        module = __import__(path, globals(), locals(), [name], level)
        klass = getattr(module, name)
        instance = klass(language=self.language, method_name=self.method_name,
                         class_name=self.class_name)
        result = instance.port(model)
        if self.with_details:
            return self.get_details(result)
        return result

    def get_details(self, model):
        """
        Get additional and useful information.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return data : dict
            language : string
                The target programming language.
            filename : string
                A valid filename.
            compiling_cmd : string
                The command to compile the ported model.
            execution_cmd : string
                The command to execute the ported model (after compiling).
            model : string
                The ported model.
        """

        # Filename:
        filename = self.class_name.lower()
        if self.language == 'java':
            filename = filename.capitalize()
        filename = '%s.%s' % (filename, self.language)

        # Commands:
        comp_cmd = ''  # compiling command
        exec_cmd = ''  # execution command
        if self.language is 'c':
            class_name = self.class_name.lower()
            # gcc tmp.c -o tmp
            comp_cmd = 'gcc %s -o %s' % (filename, class_name)
            # ./tmp
            exec_cmd = os.path.join('.', class_name)
        elif self.language is 'java':
            # javac Tmp.java
            comp_cmd = 'javac %s' % filename
            # java -classpath . Tmp
            class_name = self.class_name.capitalize()
            exec_cmd = 'java -classpath . %s' % class_name
        elif self.language is 'js':
            # node tmp.js
            exec_cmd = 'node %s' % filename
        elif self.language is 'go':
            # TODO: Add go-relevant commands
            pass

        # Result:
        result = {
            'language': self.language,
            'filename': filename,
            'compiling_cmd': comp_cmd,
            'execution_cmd': exec_cmd,
            'model': model,
        }
        return result

    @staticmethod
    def get_model_data(model):
        """
        Get data of the assigned model.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return cat : string ('regressor', 'classifier')
            The model category.

        :return name : string
            The name of the used algorithm.
        """
        cat = Porter.is_supported_model(model)
        name = type(model).__name__
        return cat, name

    @staticmethod
    def is_supported_classifier(model):
        """
        Check whether the model is a convertible classifier.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return : bool
            Whether the model is a supported classifier.
        """

        # Default classifiers:
        supported_clfs = (
            DecisionTreeClassifier,
            AdaBoostClassifier,
            RandomForestClassifier,
            ExtraTreesClassifier,
            LinearSVC,
            SVC,
            NuSVC,
            KNeighborsClassifier,
            GaussianNB,
            BernoulliNB,
        )

        # Get version information:
        import sklearn
        version = sklearn.__version__.split('.')
        major, minor = int(version[0]), int(version[1])

        # MLPClassifier was new in version 0.18:
        if major > 0 or (major == 0 and minor >= 18):  # version >= 0.18
            from sklearn.neural_network.multilayer_perceptron import \
                MLPClassifier
            supported_clfs += (MLPClassifier, )

        return isinstance(model, supported_clfs)

    @staticmethod
    def is_supported_regressor(model):
        """
        Check whether the model is a convertible classifier.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return : bool
            Whether the model is a supported regressor.
        """
        # return isinstance(model, ())
        return False

    @staticmethod
    def is_supported_model(model):
        """
        Check whether the model is a convertible classifier or regressor.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return : bool
            Whether the model is a supported model.

        See also
        --------
        onl.nok.sklearn.classifier.*, onl.nok.sklearn.regressor.*
        """

        # Is the model a supported classifier?
        if Porter.is_supported_classifier(model):
            return 'classifier'

        # Is the model a supported regressor?
        if Porter.is_supported_regressor(model):
            return 'regressor'

        msg = 'The model is not an instance of '\
              'a supported classifier or regressor.'
        raise ValueError(msg)
