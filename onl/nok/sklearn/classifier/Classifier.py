

class Classifier(object):

    SUPPORT = {}
    TEMPLATE = {}

    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name

        if method_name not in self.__class__.SUPPORT:
            msg = 'The given method is not supported by the chosen classifier.'
            raise AttributeError(msg)
        else:
            if self.language not in self.__class__.SUPPORT[method_name]:
                msg = ('The given programming language is '
                       'not supported for the given method.')
                raise AttributeError(msg)


    def temp(self, name, templates=None):
        """Get specific template of chosen programming language.

        Parameters
        ----------
        :param name : string
            The name of the template.

        Returns
        -------
        :return : string
            The template string.
        """
        if templates is None:
            templates = self.TEMPLATE.get(self.language)
        keys = name.split('.')
        key = keys.pop(0).lower()
        template = templates.get(key, None)
        if template is not None:
            if type(template) is str:
                return template
            else:
                keys = '.'.join(keys)
                return self.temp(keys, template)
        else:
            raise AttributeError('Template not found')


    def port(self, model):
        """Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        self.model = model
