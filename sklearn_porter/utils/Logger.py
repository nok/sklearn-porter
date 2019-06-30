# -*- coding: utf-8 -*-

from pathlib import Path

import logging
from logging.config import fileConfig


class Logger:

    loggers = {}

    @staticmethod
    def get_logger(name: str = '') -> logging.Logger:
        if name not in Logger.loggers.keys():
            config_path = Path(__file__).parent / 'logging.ini'
            config_path = str(config_path)  # for Python 3.5
            fileConfig(config_path)
            Logger.loggers[name] = logging.getLogger(name)
        return Logger.loggers.get(name)

    @staticmethod
    def set_level(level: int):
        for name, logger in Logger.loggers.items():
            logger.setLevel(level)


def get_logger(name: str = '') -> logging.Logger:
    return Logger.get_logger(name)


def set_level(level: int):
    Logger.set_level(level)
