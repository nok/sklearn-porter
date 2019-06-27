# -*- coding: utf-8 -*-

from os.path import sep

from sklearn_porter.language.LanguageABC import LanguageABC


class Ruby(LanguageABC):
    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    TEMP_DIR = 'ruby'
    SUFFIX = 'rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = 'ruby {dest_dir}' + sep + '{dest_file}'

    TEMPLATES = {
        'init':         '{name} = {value}',

        # if/else condition:
        'if':           'if {0} {1} {2}',
        'else':         'else',
        'endif':        'end',

        # Basics:
        'indent':       '    ',
        'join':         ' ',
        'type':         '{0}',

        # Arrays:
        'in_brackets':  '[{0}]',
        'arr[]':        '{name} = [{values}]',  # ages = [1, 2]
        'arr[][]':      '{name} = [{values}]',
        'arr[][][]':    '{name} = [{values}]',

        # Primitive data types:
        'int':          '',
        'double':       ''
    }
