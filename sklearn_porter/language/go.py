# -*- coding: utf-8 -*-

from os.path import sep


KEY = 'go'
LABEL = 'Go'

DEPENDENCIES = ['go']
TEMP_DIR = 'go'
SUFFIX = 'go'

# go build -o tmp/estimator tmp/estimator.go
CMD_COMPILE = 'go build -o {dest_dir}' + sep + '{dest_file} {src_dir}' + sep + '{src_file}'

# tmp/estimator <args>
CMD_EXECUTE = '{dest_dir}' + sep + '{dest_file}'

TEMPLATES = {
    # if/else condition:
    'if':           'if {0} {1} {2} {{',
    'else':         '} else {',
    'endif':        '}',

    # Basics:
    'indent':       '\t',
    'join':         '',
    'type':         '{0}',

    # Arrays:
    'in_brackets':  '{{{0}}}',
    'arr[]':        '{name} := []{type} {{{values}}}',  # ages := []int {1, 2}
    'arr[][]':      '{name} := [][]{type} {{{values}}}',

    # Primitive data types:
    'int':          'int',
    'double':       'float64'
}
