# -*- coding: utf-8 -*-

import unittest

from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class DecisionTreeClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        mdl = DecisionTreeClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
