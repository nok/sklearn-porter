# -*- coding: utf-8 -*-


class Go(object):

    KEY = 'go'
    LABEL = 'Go'

    DEPENDENCIES = ['go']
    TEMP_DIR = 'go'
    SUFFIX = '.go'

    # go build -o tmp/estimator tmp/estimator.go
    CMD_COMPILE = 'go build -o {dest_dir}/{dest_file} {src_dir}/{src_file}'

    # tmp/estimator <args>
    CMD_EXECUTE = '{dest_dir}/{dest_file}'
