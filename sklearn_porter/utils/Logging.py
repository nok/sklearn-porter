# -*- coding: utf-8 -*-

import logging


LOGGING_FORMAT = '%(name)-12s: %(levelname)-8s %(message)s'


class Logging(object):

    @staticmethod
    def get_logger(name, level=0):
        """Setup a logging instance"""
        console = logging.StreamHandler()
        level = [logging.NOTSET, logging.ERROR, logging.WARN, logging.INFO,
                 logging.DEBUG][level]
        console.setLevel(level)
        formatter = logging.Formatter(LOGGING_FORMAT)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        return logging.getLogger(name)
