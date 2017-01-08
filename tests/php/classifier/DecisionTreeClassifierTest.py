# -*- coding: utf-8 -*-

import unittest

from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter

from ..PhpTest import PhpTest


class DecisionTreeClassifierTest(PhpTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.porter = Porter(language='php')
        clf = DecisionTreeClassifier(random_state=0)
        self._port_model(clf)

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
