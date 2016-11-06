import os.path


class Classifier(object):

    SUPPORT = {}
    TEMPLATE = {}


    def __init__(
            self, language='java', method_name='predict', class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name
        self.check_support()


    def check_support(self):
        """Check template and programming language support."""
        if self.method_name not in self.__class__.SUPPORT:
            msg = ('The given method is not supported '
                   'by the chosen classifier.')
            raise AttributeError(msg)
        else:
            if self.language not in self.__class__.SUPPORT[self.method_name]:
                msg = ('The given programming language is '
                       'not supported for the given method.')
                raise AttributeError(msg)


    def indent(self, text, indentation=4, skipping=False):
        """
        Indent text with single spaces.

        Parameters
        ----------
        :param text : string
            The text which get a specific indentation.
        :param indentation : int
            The number of indentations.
        :param skipping : boolean
            Whether to skip the initial indentation.

        Returns
        -------
        :return : string
            The indented text.
        """
        lines = text.splitlines()

        # TODO: Refactor the indentation type:
        space = self.TEMPLATE.get(self.language).get('indent', ' ')

        # Single line:
        if len(lines) == 1:
            if skipping:
                return text.strip()
            return indentation * space + text.strip()

        # Multiple lines:
        indented_lines = []
        for idx, line in enumerate(lines):
            if skipping and idx is 0:
                indented_lines.append(line)
            else:
                line = indentation * space + line
                indented_lines.append(line)
        indented_text = '\n'.join(indented_lines)
        return indented_text


    def temp(self, name, templates=None, indentation=None, skipping=False):
        """
        Get specific template of chosen programming language.

        Parameters
        ----------
        :param name : string
            The key name of the template.
        :param tempaltes : string
            The template with placeholders.
        :param indentation : int
            The number of indentations.
        :param skipping : boolean
            Whether to skip the initial indentation.

        Returns
        -------
        :return : string
            The wanted template string.
        """
        if templates is None:
            templates = self.TEMPLATE.get(self.language)
        keys = name.split('.')
        key = keys.pop(0).lower()
        template = templates.get(key, None)
        if template is not None:
            if type(template) is str:
                if indentation is not None:
                    template = self.indent(template, indentation, skipping)
                return template
            else:
                keys = '.'.join(keys)
                return self.temp(keys, template, skipping=False)
        else:
            path = os.path.join(
                os.path.dirname(__file__), self.__class__.__name__,
                'templates', self.language, name + '.txt')
            if os.path.isfile(path):
                template = open(path, 'r').read()
                if indentation is not None:
                    template = self.indent(template, indentation, skipping)
                return template
            else:
                raise AttributeError('Template \'%s\' not found.' % (name))


    def port(self, model):
        """
        Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        self.model = model
