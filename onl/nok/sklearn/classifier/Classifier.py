

class Classifier(object):

    SUPPORT = {}

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name

        if method_name not in self.__class__.SUPPORT:
            msg = 'The given method is not supported by the chosen classifier.'
            raise AttributeError(msg)
        else:
            if self.language not in self.__class__.SUPPORT[method_name]:
                msg = 'The given programming language is not supported for the given method.'
                raise AttributeError(msg)


    def port(self, model):
        """Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        self.model = model
