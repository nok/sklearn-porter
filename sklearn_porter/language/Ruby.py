# -*- coding: utf-8 -*-

from os.path import sep


class Ruby(object):

    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    TEMP_DIR = 'ruby'
    SUFFIX = '.rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = 'ruby {dest_dir}' + sep + '{dest_file}'
