# -*- coding: utf-8 -*-

from enum import Enum

from sklearn_porter.language import *

import sklearn_porter.exceptions


class Method(Enum):
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'

    @classmethod
    def convert(cls, method):
        if method and isinstance(method, str):
            try:
                method = Method[method.upper()]
            except KeyError:
                raise sklearn_porter.exceptions.InvalidMethodError(method)
        return method


ALL_METHODS = {
    Method.PREDICT,
    Method.PREDICT_PROBA,
}


class Template(Enum):
    ATTACHED = 'attached'
    COMBINED = 'combined'
    EXPORTED = 'exported'

    @classmethod
    def convert(cls, template):
        if template and isinstance(template, str):
            try:
                template = Template[template.upper()]
            except KeyError:
                raise sklearn_porter.exceptions.InvalidTemplateError(template)
        return template


ALL_TEMPLATES = {
    Template.ATTACHED,
    Template.COMBINED,
    Template.EXPORTED,
}


class Language(Enum):
    C = C
    GO = Go
    JAVA = Java
    JS = JavaScript
    PHP = PHP
    RUBY = Ruby

    @classmethod
    def convert(cls, language):
        if language and isinstance(language, str):
            try:
                language = Language[language.upper()]
            except KeyError:
                raise sklearn_porter.exceptions.InvalidLanguageError(language)
        return language


ALL_LANGUAGES = {
    Language.C,
    Language.GO,
    Language.JAVA,
    Language.JS,
    Language.PHP,
    Language.RUBY,
}
