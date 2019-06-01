# -*- coding: utf-8 -*-

from sklearn_porter.utils.Options import Options, set_option
from sklearn_porter.utils.Logger import Logger, get_logger


def get_qualname(obj: object):
    return obj.__class__.__module__ + '.' + obj.__class__.__qualname__
