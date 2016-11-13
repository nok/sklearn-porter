import os


class Porter():

    __version__ = '0.1.0'

    def __init__(
            self, language="java", method_name='predict',
            class_name='Tmp', with_details=False):
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
        :return: string
            The ported model as string.
        """
        md_type, md_name = self.get_model_data(model)
        md_path = '.'.join([md_type, md_name])
        md_mod = __import__(md_path, globals(), locals(), [md_name], -1)
        klass = getattr(md_mod, md_name)
        instance = klass(
            language=self.language,
            method_name=self.method_name,
            class_name=self.class_name)
        ported_model = instance.port(model)
        if self.with_details:
            return self.get_details(ported_model)
        return ported_model


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
        filename = '%s.%s' % (self.class_name.lower(), self.language)
        if self.language == 'java':
            filename = filename.capitalize()

        comp_cmd = ''  # compiling command
        exec_cmd = ''  # execution command
        if self.language is 'c':
            # gcc tmp.c -o tmp
            comp_cmd = 'gcc %s -o %s' % (filename, self.class_name.lower())
            # ./tmp
            exec_cmd = os.path.join('.', self.class_name.lower())
        elif self.language is 'java':
            # javac Tmp.java
            comp_cmd = 'javac %s' % filename
            # java -classpath . Tmp
            exec_cmd = 'java -classpath . %s' % self.class_name.capitalize()
        elif self.language is 'js':
            # node tmp.js
            exec_cmd = 'node %s' % filename
        elif self.language is 'go':
            pass

        data = {
            'language': self.language,
            'filename': filename,
            'compiling_cmd': comp_cmd,
            'execution_cmd': exec_cmd,
            'model': model,
        }
        return data


    def get_model_data(self, model):
        """
        Get data of the assigned model.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return md_type : string ['regressor', 'classifier']
            The model type.

        :return md_name : string
            The name of the used algorithm.
        """
        md_type = self.is_supported_model(model)
        md_name = type(model).__name__
        return md_type, md_name


    def is_supported_classifier(self, model):
        """
        Check whether the model is a convertible classifier.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return bool
            Whether the model is a supported classifier.
        """
        path = model.__class__.__module__ + '.' + model.__class__.__name__
        return path in [
            'sklearn.neural_network.multilayer_perceptron.MLPClassifier',
            'sklearn.tree.tree.DecisionTreeClassifier',
            'sklearn.ensemble.weight_boosting.AdaBoostClassifier',
            'sklearn.ensemble.forest.RandomForestClassifier',
            'sklearn.ensemble.forest.ExtraTreesClassifier',
            'sklearn.svm.classes.LinearSVC',
            'sklearn.svm.classes.SVC',
        ]


    def is_supported_regressor(self, model):
        """
        Check whether the model is a convertible classifier.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return bool
            Whether the model is a supported regressor.
        """
        path = model.__class__.__module__ + '.' + model.__class__.__name__
        return path in []


    def is_supported_model(self, model):
        """
        Check whether the model is a convertible classifier or regressor.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        Returns
        -------
        :return bool
            Whether the model is a supported model.

        See also
        --------
        onl.nok.sklearn.classifier.*, onl.nok.sklearn.regressor.*
        """
        if self.is_supported_classifier(model):
            return 'classifier'
        if self.is_supported_regressor(model):
            return 'regressors'
        msg = 'The model is not an instance of '\
              'a supported classifier or regressor.'
        raise ValueError(msg)
