from enum import Enum

from sklearn_porter import exceptions as exception
from sklearn_porter.language import (  # noqa: F401 (re-export LANGUAGES)
    LANGUAGES,
    PHP,
    C,
    CSharp,
    Go,
    Java,
    JavaScript,
    Ruby,
)


class Method(Enum):
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'

    @classmethod
    def convert(cls, method):
        if method and isinstance(method, str):
            try:
                method = Method[method.upper()]
            except KeyError:
                raise exception.InvalidMethodError(method) from None
        return method


ALL_METHODS = {Method.PREDICT, Method.PREDICT_PROBA}


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
                raise exception.InvalidTemplateError(template) from None
        return template


class Language(Enum):
    C = C
    CS = CSharp
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
                raise exception.InvalidLanguageError(language) from None
        return language
