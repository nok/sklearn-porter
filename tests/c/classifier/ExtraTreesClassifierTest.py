# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter

from ..CTest import CTest


class ExtraTreesClassifierTest(CTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.porter = Porter(language='c')
        self._port_model(ExtraTreesClassifier(random_state=0))

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
