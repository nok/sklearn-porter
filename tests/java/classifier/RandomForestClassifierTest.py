# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class RandomForestClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(RandomForestClassifierTest, self).setUp()
        self.porter = Porter(language='java')
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self._port_model(clf)

    def tearDown(self):
        super(RandomForestClassifierTest, self).tearDown()
