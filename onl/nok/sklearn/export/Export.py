class Export:
    """Base class for model or prediction porting."""


    @staticmethod
    def export(model, method_name='predict', class_name='Tmp'):
        """Export the prediction of a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : Model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        :param method_name : String (default='predict')
            The name of the prediction method.

        :param class_name : String (default='Tmp')
            The name of the environment class.
        """

        model_type = Export.is_convertible_model(model)
        if not model_type:
            return

        model_name = type(model).__name__
        model_path = '.'.join(['onl.nok.sklearn.export', model_type, model_name])

        import os
        import sys
        sys.path.append(os.getcwd())

        module = __import__(model_path, globals(), locals(), [model_name], -1)
        model_klass = getattr(module, model_name)
        model_method = getattr(model_klass, 'export')
        return model_method(model, method_name=method_name, class_name=class_name)


    @staticmethod
    def get_convertible_classifiers():
        '''Get a list of convertible classifiers.'''
        import sklearn
        from sklearn.ensemble import weight_boosting
        return [
            sklearn.tree.tree.DecisionTreeClassifier,
            sklearn.ensemble.AdaBoostClassifier
        ]


    @staticmethod
    def get_convertible_regressors():
        '''Get a list of all convertible regressors.'''
        return []


    @staticmethod
    def is_convertible_model(model):
        """Check whether the model is a convertible classifier or regressor.

        Parameters
        ----------
        :param model : Model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        See also
        --------
        onl.nok.sklearn.export.classifier.*, onl.nok.sklearn.export.regressor.*
        """

        classifiers = Export.get_convertible_classifiers()
        is_convertible_clf = type(model) in classifiers
        if is_convertible_clf:
            return 'classifier'

        regressors = Export.get_convertible_regressors()
        is_convertible_rgs = type(model) in regressors
        if is_convertible_rgs:
            return 'regressors'

        if not is_convertible_clf and not is_convertible_rgs:
            raise ValueError('The model is not an instance of a supported classifier or regressor.')
        return False


def main():
    import sys

    # TODO: In general add more error exceptions
    if len(sys.argv) == 3:
        import os

        is_valid_input_file = lambda f: str(f).endswith('.pkl') and os.path.isfile(str(f))
        is_valid_output_file = lambda f: str(f).endswith('.java') or str(f).endswith('.c')

        if is_valid_input_file(sys.argv[1]) and is_valid_output_file(sys.argv[2]):
            input_file = str(sys.argv[1])
            output_file = str(sys.argv[2])

            from sklearn.externals import joblib
            with open(output_file, 'w') as file:
                model = joblib.load(input_file)
                class_name = output_file.split('.')[-2].lower().title()
                file.write(Export.export(model, class_name=class_name))


if __name__ == '__main__':
    main()