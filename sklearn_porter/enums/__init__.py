# -*- coding: utf-8 -*-

from enum import Enum

from sklearn_porter.language import *


class Method(Enum):
    PREDICT = 'predict'
    PREDICT_PROBA = 'predict_proba'


class Template(Enum):
    COMBINED = 'combined'
    ATTACHED = 'attached'
    EXPORTED = 'exported'


class Language(Enum):
    C = C
    GO = Go
    JAVA = Java
    JS = JavaScript
    PHP = PHP
    RUBY = Ruby
