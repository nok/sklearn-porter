# -*- coding: utf-8 -*-

from os.path import sep


KEY = 'java'
LABEL = 'Java'

DEPENDENCIES = ['java', 'javac']
TEMP_DIR = 'java'
SUFFIX = '.java'

# javac {class_path} tmp/Estimator.java
# class_path = '-cp ./gson.jar'
CMD_COMPILE = 'javac {class_path} {src_dir}' + sep + '{src_file}'

# java {class_path} Estimator <args>
# class_path = '-cp ./gson.jar:./tmp'
CMD_EXECUTE = 'java {class_path} {dest_dir}' + sep + '{dest_file}'

TEMPLATES = {
    # if/else condition:
    'if':           'if ({0} {1} {2}) {{',
    'else':         '} else {',
    'endif':        '}',

    # Basics:
    'indent':       '    ',
    'join':         '; ',
    'type':         '{0}',

    # Arrays:
    'in_brackets':  '{{{0}}}',
    'arr[]':        '{type}[] {name} = {{{values}}};',  # int[] ages = {1, 2};
    'arr[][]':      '{type}[][] {name} = {{{values}}};',

    # Primitive data types:
    'int':          'int',
    'double':       'double'
}
