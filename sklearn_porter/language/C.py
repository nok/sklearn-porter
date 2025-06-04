# -*- coding: utf-8 -*-


class C(object):

    KEY = 'c'
    LABEL = 'C'

    DEPENDENCIES = ['gcc']
    TEMP_DIR = 'c'
    SUFFIX = '.c'

    # gcc tmp/estimator.c -std=c99 -lm -o tmp/estimator
    CMD_COMPILE = 'gcc {src_dir}/{src_file} -std=c99 -lm -o {dest_dir}/{dest_file}'

    # tmp/estimator <args>
    CMD_EXECUTE = '{dest_dir}/{dest_file}'
