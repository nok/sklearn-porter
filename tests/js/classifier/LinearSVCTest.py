# -*- coding: utf-8 -*-

import unittest

from sklearn.svm.classes import LinearSVC
from sklearn_porter import Porter

from ..JavaScriptTest import JavaScriptTest


class LinearSVCTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.porter = Porter(language='js')
        clf = LinearSVC(C=1., random_state=0)
        self._port_model(clf)

    def tearDown(self):
        super(LinearSVCTest, self).tearDown()
