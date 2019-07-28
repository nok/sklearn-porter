# -*- coding: utf-8 -*-

from sklearn_porter.language.LanguageABC import LanguageABC


class Ruby(LanguageABC):
    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    TEMP_DIR = 'ruby'
    SUFFIX = 'rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = 'ruby {src_path}'

    TEMPLATES = {
        'init':         '{name} = {value}',

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
        'arr[]':        '{{ name }} = [{{ values }}]',  # ages = [1, 2]
        'arr[][]':      '{{ name }} = [{{ values }}]',
        'arr[][][]':    '{{ name }} = [{{ values }}]',

        # Primitive data types:
        'int':          '',
        'double':       ''
    }
