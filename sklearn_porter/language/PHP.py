# -*- coding: utf-8 -*-


class PHP(object):

    KEY = 'php'
    LABEL = 'PHP'

    DEPENDENCIES = ['php']
    TEMP_DIR = 'php'
    SUFFIX = '.php'

    CMD_COMPILE = None

    # php -f {} Estimator.php <args>
    CMD_EXECUTE = 'php -f {dest_dir}/{dest_file}'
