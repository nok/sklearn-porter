# -*- coding: utf-8 -*-

from collections import OrderedDict

from sklearn_porter.language.c import C
from sklearn_porter.language.go import Go
from sklearn_porter.language.java import Java
from sklearn_porter.language.js import JavaScript
from sklearn_porter.language.php import PHP
from sklearn_porter.language.ruby import Ruby

LANGUAGES = OrderedDict(
    {
        C.KEY: C,
        Go.KEY: Go,
        Java.KEY: Java,
        JavaScript.KEY: JavaScript,
        PHP.KEY: PHP,
        Ruby.KEY: Ruby
    }
)

__all__ = ['C', 'Go', 'Java', 'JavaScript', 'PHP', 'Ruby', 'LANGUAGES']
