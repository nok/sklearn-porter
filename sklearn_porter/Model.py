# -*- coding: utf-8 -*-

import os.path


class Model(object):

    SUPPORTED_METHODS = {}
    TEMPLATES = {}

    def __init__(self, language='java', method_name='predict',
                 class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name
        self.check_support()

    def check_support(self):
        """Check template and programming language support."""

        if self.method_name not in self.__class__.SUPPORTED_METHODS:
            err_msg = ('The given method is not supported '
                       'by the chosen classifier.')
            raise AttributeError(err_msg)

        if self.language not in self.__class__.TEMPLATES.keys():
            err_msg = ('The given programming language is '
                       'not supported for the given method.')
            raise AttributeError(err_msg)

    def indent(self, text, n_indents=1, skipping=False):
        """
        Indent text with single spaces.

        Parameters
        ----------
        :param text : string
            The text which get a specific indentation.
        :param n_indents : int
            The number of indentations.
        :param skipping : boolean
            Whether to skip the initial indentation.

        Returns
        -------
        :return : string
            The indented text.
        """
        lines = text.splitlines()
        space = self.TEMPLATES.get(self.language).get('indent', ' ')

        # Single line:
        if len(lines) == 1:
            if skipping:
                return text.strip()
            return n_indents * space + text.strip()

        # Multiple lines:
        indented_lines = []
        for idx, line in enumerate(lines):
            if skipping and idx is 0:
                indented_lines.append(line)
            else:
                line = n_indents * space + line
                indented_lines.append(line)
        indented_text = '\n'.join(indented_lines)
        return indented_text

    def temp(self, name, templates=None, n_indents=None, skipping=False):
        """
        Get specific template of chosen programming language.

        Parameters
        ----------
        :param name : string
            The key name of the template.
        :param tempaltes : string
            The template with placeholders.
        :param n_indents : int
            The number of indentations.
        :param skipping : boolean
            Whether to skip the initial indentation.

        Returns
        -------
        :return : string
            The wanted template string.
        """
        if templates is None:
            templates = self.TEMPLATES.get(self.language)
        keys = name.split('.')
        key = keys.pop(0).lower()
        template = templates.get(key, None)
        if template is not None:
            if type(template) is str:
                if n_indents is not None:
                    template = self.indent(template, n_indents, skipping)
                return template
            else:
                keys = '.'.join(keys)
                return self.temp(keys, template, skipping=False)
        else:
            # TODO: Replace static path definition: 'classifier'
            class_name = self.__class__.__name__
            path = os.path.join(
                os.path.dirname(__file__), 'classifier', class_name,
                'templates', self.language, name + '.txt')
            if os.path.isfile(path):
                template = open(path, 'r').read()
                if n_indents is not None:
                    template = self.indent(template, n_indents, skipping)
                return template
            else:
                raise AttributeError('Template "%s" not found.' % name)

    def port(self, model):
        """
        Port a trained model in the syntax of a specific programming language.

        Parameters
        ----------
        :param model : scikit-learn model object
            An instance of a trained model (e.g. DecisionTreeClassifier).
        """
        self.model = model
