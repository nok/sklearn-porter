# -*- coding: utf-8 -*-

from os.path import sep


KEY = 'go'
LABEL = 'Go'

DEPENDENCIES = ['go']
TEMP_DIR = 'go'
SUFFIX = '.go'

# go build -o tmp/estimator tmp/estimator.go
CMD_COMPILE = 'go build -o {dest_dir}' + sep + '{dest_file} {src_dir}' + sep + '{src_file}'

# tmp/estimator <args>
CMD_EXECUTE = '{dest_dir}' + sep + '{dest_file}'
