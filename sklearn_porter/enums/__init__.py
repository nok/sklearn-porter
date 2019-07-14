# -*- coding: utf-8 -*-

from enum import Enum

from sklearn_porter.language import *


class Method(Enum):
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'


ALL_METHODS = {
    Method.PREDICT,
    Method.PREDICT_PROBA,
}


class Template(Enum):
    ATTACHED = 'attached'
    COMBINED = 'combined'
    EXPORTED = 'exported'


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


ALL_LANGUAGES = {
    Language.C,
    Language.GO,
    Language.JAVA,
    Language.JS,
    Language.PHP,
    Language.RUBY,
}
