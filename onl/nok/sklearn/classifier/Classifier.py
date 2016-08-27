

class Classifier(object):

    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name


    def port(self, model):
        """Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        self.model = model
