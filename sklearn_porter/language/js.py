# -*- coding: utf-8 -*-

from os.path import sep


KEY = 'js'
LABEL = 'JavaScript'

DEPENDENCIES = ['node']
TEMP_DIR = 'js'
SUFFIX = 'js'

CMD_COMPILE = None

# node estimator.js <args>
CMD_EXECUTE = 'node {dest_dir}' + sep + '{dest_file}'
