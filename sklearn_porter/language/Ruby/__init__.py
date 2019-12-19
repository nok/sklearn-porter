# -*- coding: utf-8 -*-

from sklearn_porter.language.LanguageABC import LanguageABC


class Ruby(LanguageABC):
    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    SUFFIX = 'rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = 'ruby {src_path}'

    # yapf: disable
    TEMPLATES = {
        'init':         '{{ name }} = {{ value }}',

        # if/else condition:
        'if':           'if {{ a }} {{ op }} {{ b }}',
        'else':         'else',
        'endif':        'end',

        # Basics:
        'indent':       '    ',
        'join':         ' ',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '[{{ value }}]',
        'arr[]':        '{{ name }} = [{{ values }}]',
        'arr[][]':      '{{ name }} = [{{ values }}]',
        'arr[][][]':    '{{ name }} = [{{ values }}]',

        # Primitive data types:
        'int':          '',
        'double':       ''
    }
    # yapf: enable
