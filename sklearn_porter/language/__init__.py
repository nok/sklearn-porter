# -*- coding: utf-8 -*-

import sklearn_porter.language.c
import sklearn_porter.language.go
import sklearn_porter.language.java
import sklearn_porter.language.js
import sklearn_porter.language.php
import sklearn_porter.language.ruby

LANGUAGES = {
    c.KEY: c,
    go.KEY: go,
    java.KEY: java,
    js.KEY: js,
    php.KEY: php,
    ruby.KEY: ruby
}

__all__ = ['c', 'go', 'java', 'js', 'php', 'ruby', 'LANGUAGES']
