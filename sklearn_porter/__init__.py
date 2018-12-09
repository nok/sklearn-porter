# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

import os
import os.path

from sklearn_porter.Porter import Porter


def _read_version():
    """Read the module version from __version__.txt"""
    source_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(source_dir, '__version__.txt')
    version = open(version_file, 'r').readlines().pop()
    version = str(version).strip()
    return version


__version__ = _read_version()
