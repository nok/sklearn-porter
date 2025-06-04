# -*- coding: utf-8 -*-


class Ruby(object):

    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    TEMP_DIR = 'ruby'
    SUFFIX = '.rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = 'ruby {dest_dir}/{dest_file}'
