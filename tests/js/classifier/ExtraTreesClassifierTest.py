# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter

from ..JavaScriptTest import JavaScriptTest


class ExtraTreesClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        mdl = ExtraTreesClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
