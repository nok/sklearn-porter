# -*- coding: utf-8 -*-

from sklearn_porter.language.C import C
from sklearn_porter.language.Go import Go
from sklearn_porter.language.Java import Java
from sklearn_porter.language.JavaScript import JavaScript
from sklearn_porter.language.PHP import PHP
from sklearn_porter.language.Ruby import Ruby

LANGUAGES = {
    C.KEY: C,
    Go.KEY: Go,
    Java.KEY: Java,
    JavaScript.KEY: JavaScript,
    PHP.KEY: PHP,
    Ruby.KEY: Ruby
}

__all__ = ['C', 'Go', 'Java', 'JavaScript', 'PHP', 'Ruby', 'LANGUAGES']
