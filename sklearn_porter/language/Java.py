# -*- coding: utf-8 -*-


class Java(object):

    KEY = 'java'
    LABEL = 'Java'

    DEPENDENCIES = ['java', 'javac']
    TEMP_DIR = 'java'
    SUFFIX = '.java'

    # javac {class_path} tmp/Estimator.java
    # class_path = '-cp ./gson.jar'
    CMD_COMPILE = 'javac {class_path} {src_dir}/{src_file}'

    # java {class_path} Estimator <args>
    # class_path = '-cp ./gson.jar:./tmp'
    CMD_EXECUTE = 'java {class_path} {dest_dir}/{dest_file}'
