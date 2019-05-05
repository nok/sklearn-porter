# -*- coding: utf-8 -*-

from typing import Union
from pathlib import Path

from logging import Logger, config, getLogger
from logging import CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET


def get_logger(name: str, logger: Union[Logger, int]) -> Logger:
    if isinstance(logger, Logger):
        logger.name = name
        return logger

    level = ERROR
    if logger in (CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET):
        level = logger

    config_path = Path(__file__).parent / 'logging.ini'
    config.fileConfig(config_path)
    logger = getLogger(name)
    logger.setLevel(level)

    return logger
