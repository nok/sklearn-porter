# -*- coding: utf-8 -*-


class JavaScript(object):

    KEY = 'js'
    LABEL = 'JavaScript'

    DEPENDENCIES = ['node']
    TEMP_DIR = 'js'
    SUFFIX = '.js'

    CMD_COMPILE = None

    # node estimator.js <args>
    CMD_EXECUTE = 'node {dest_dir}/{dest_file}'
