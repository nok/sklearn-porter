# -*- coding: utf-8 -*-

from sklearn_porter.utils.logging import get_logger
from sklearn_porter.utils.mapping import get_mapper


def get_qualname(obj: object):
    return obj.__class__.__module__ + '.' + obj.__class__.__qualname__


__all__ = [
    'get_logger',
    'get_mapper',
    'get_qualname'
]
