# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

import os.path


class Template(object):

    SUPPORTED_METHODS = {}
    TEMPLATES = {}

    def __init__(self, estimator, target_language='java',
                 target_method='predict', **kwargs):
        # pylint: disable=unused-argument
        self.target_language = str(target_language)
        self.target_method = str(target_method)

        # Default settings:
        self.class_name = 'Brain'
        self.method_name = 'predict'
        self._num_format = lambda x: str(x)
        self.use_file = False

    def indent(self, text, n_indents=1, skipping=False):
        """
        Indent text with single spaces.

        Parameters
        ----------
        :param text : string
            The text which get a specific indentation.
        :param n_indents : int, default: 1
            The number of indentations.
        :param skipping : boolean, default: False
            Whether to skip the initial indentation.

        Returns
        -------
        return : string
            The indented text.
        """
        lines = text.splitlines()
        space = self.TEMPLATES.get(self.target_language).get('indent', ' ')

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
        Get specific template of chosen
        programming language.

        Parameters
        ----------
        :param param name : string
            The key name of the template.
        :param param templates : string, default: None
            The template with placeholders.
        :param param n_indents : int, default: None
            The number of indentations.
        :param param skipping : bool, default: False
            Whether to skip the initial indentation.

        Returns
        -------
        return : string
            The wanted template string.
        """
        if templates is None:
            templates = self.TEMPLATES.get(self.target_language)
        keys = name.split('.')
        key = keys.pop(0).lower()
        template = templates.get(key, None)
        if template is not None:
            if isinstance(template, str):
                if n_indents is not None:
                    template = self.indent(template, n_indents, skipping)
                return template
            else:
                keys = '.'.join(keys)
                return self.temp(keys, template, skipping=False)
        else:
            class_name = self.__class__.__name__
            estimator_type = getattr(self, 'estimator_type') if \
                hasattr(self, 'estimator_type') else 'classifier'
            path = os.path.join(os.path.dirname(__file__), 'estimator',
                                estimator_type, class_name, 'templates',
                                self.target_language, name + '.txt')
            if os.path.isfile(path):
                with open(path, 'r') as file_:
                    template = file_.read()
                if n_indents is not None:
                    template = self.indent(template, n_indents, skipping)
                return template
            else:
                err = "Template '{}' wasn't found.".format(name)
                raise AttributeError(err)

    def repr(self, value):
        return self._num_format(value)

    def data(self, dict_):
        copy = self.__dict__.copy()
        copy.update(dict_)  # update and extend dictionary
        return copy
