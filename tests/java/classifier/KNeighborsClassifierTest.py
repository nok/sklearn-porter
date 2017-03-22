# -*- coding: utf-8 -*-

import unittest

from sklearn.neighbors import KNeighborsClassifier
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class KNeighborsClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(KNeighborsClassifierTest, self).setUp()
        mdl = KNeighborsClassifier(algorithm='brute', n_neighbors=3, weights='uniform')
        self._port_model(mdl)

    def tearDown(self):
        super(KNeighborsClassifierTest, self).tearDown()
