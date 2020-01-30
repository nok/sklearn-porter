# -*- coding: utf-8 -*-

from sklearn_porter.language.LanguageABC import LanguageABC


class C(LanguageABC):
    KEY = 'c'
    LABEL = 'C'

    DEPENDENCIES = ['gcc']
    SUFFIX = 'c'

    # gcc tmp/estimator.c -std=c99 -lm -o tmp/estimator
    CMD_COMPILE = 'gcc {src_path} -std=c99 -lm -o {dest_path}'

    # tmp/estimator <args>
    CMD_EXECUTE = '{dest_path}'

    # yapf: disable
    TEMPLATES = {
        'init':         '{{ type }} {{ name }} = {{ value }};',

        # if/else condition:
        'if':           'if ({{ a }} {{ op }} {{ b }}) {',
        'else':         '} else {',
        'endif':        '}',

        # Basics:
        'indent':       '    ',
        'join':         '; ',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '{{ "{" }}{{ value }}{{ "}" }}',
        # in ages[2] = {1, 2};
        'arr[]':        '{{ type }} {{ name }}[{{ n }}] = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long
        'arr[][]':      '{{ type }} {{ name }}[{{ n }}][{{ m }}] = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long
        'arr[][][]':    '{{ type }} {{ name }}[{{ n }}][{{ m }}][{{ k }}] = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long

        # Primitive data types:
        'int':          'int',
        'double':       'double'
    }
    # yapf: enable
