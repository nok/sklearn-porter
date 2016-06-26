import sklearn


class Export:
    """Base class for model or prediction porting."""

    @staticmethod
    def export(model, method='predict', class_name=None, with_env=False):
        """Export the prediction of a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : Model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        :param method : String (default='predict')
            The name of the prediction method.
        """

        model_type = Export._is_convertible_model(model)
        if not model_type:
            return
        model_name = type(model).__name__
        path = 'onl.nok.sklearn.export.' + model_type + '.' + model_name
        class_ = getattr(__import__(path, fromlist=[model_name]), model_name)
        return class_.export(model, method, with_env=with_env)

    @staticmethod
    def _get_convertible_classifiers():
        '''Get a list of convertible classifiers.'''
        return [
            sklearn.tree.tree.DecisionTreeClassifier
        ]

    @staticmethod
    def _get_convertible_regressors():
        '''Get a list of all convertible regressors.'''
        return []

    @staticmethod
    def _is_convertible_model(model):
        """Check whether the model is a convertible classifier or regressor.

        Parameters
        ----------
        :param model : Model object
            An instance of a trained model (e.g. DecisionTreeClassifier()).

        See also
        --------
        onl.nok.sklearn.export.classifier.*, onl.nok.sklearn.export.regressor.*
        """

        classifiers = Export._get_convertible_classifiers()
        is_convertible_clf = any(isinstance(model, e) for e in classifiers)
        if is_convertible_clf:
            return 'classifier'

        regressors = Export._get_convertible_regressors()
        is_convertible_rgs = any(isinstance(model, e) for e in regressors)
        if is_convertible_rgs:
            return 'regressors'

        if not is_convertible_clf and not is_convertible_rgs:
            raise ValueError('The model is not an instance of a supported classifier or regressor.')
        return False

def main():
    import sys
    import os

    if len(sys.argv) == 3:
        is_valid_input_file = lambda f: str(f).endswith('.pkl') and os.path.isfile(str(f))
        is_valid_output_file = lambda f: str(f).endswith('.java') or str(f).endswith('.c')

        if is_valid_input_file(sys.argv[1]) and is_valid_output_file(sys.argv[2]):
            input_file = str(sys.argv[1])
            output_file = str(sys.argv[2])

            from sklearn.externals import joblib
            with open(output_file, 'w') as file:
                model = joblib.load(input_file)

                class_name = output_file.split('.')[-2].lower().title()
                file.write(Export.export(model, class_name=class_name, with_env=True))

if __name__ == '__main__':
    main()