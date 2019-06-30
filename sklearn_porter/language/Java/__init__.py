# -*- coding: utf-8 -*-

from os.path import sep

from sklearn_porter.language.LanguageABC import LanguageABC


class Java(LanguageABC):
    KEY = 'java'
    LABEL = 'Java'

    DEPENDENCIES = ['java', 'javac']
    TEMP_DIR = 'java'
    SUFFIX = 'java'

    # javac {class_path} tmp/Estimator.java
    # class_path = '-cp ./gson.jar'
    CMD_COMPILE = 'javac {class_path} {src_dir}' + sep + '{src_file}'

    # java {class_path} Estimator <args>
    # class_path = '-cp ./gson.jar:./tmp'
    CMD_EXECUTE = 'java {class_path} {dest_dir}' + sep + '{dest_file}'

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
        'arr[]':        '{{ type }}[] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][]':      '{{ type }}[][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][][]':    '{{ type }}[][][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',

        # Primitive data types:
        'int':          'int',
        'double':       'double',
    }
