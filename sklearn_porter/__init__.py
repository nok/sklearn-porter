# -*- coding: utf-8 -*-

from sklearn_porter.utils import options
from sklearn_porter.Estimator import Estimator, port, make, save, test


class Version:
    """
    Basic class for semantic versioning without external dependencies.
    """
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self):
        return '.'.join(list(map(str, [self.major, self.minor, self.patch])))


__author__ = 'Darius Morawiec'
__email__ = 'nok@users.noreply.github.com'  # see https://github.com/nok/sklearn-porter
__license__ = 'MIT'
__version__ = str(Version(1, 0, 0))
