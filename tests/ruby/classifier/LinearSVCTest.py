# -*- coding: utf-8 -*-

import unittest

from sklearn.svm.classes import LinearSVC
from sklearn_porter import Porter

from ..RubyTest import RubyTest


class LinearSVCTest(RubyTest, unittest.TestCase):

    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.porter = Porter(language='ruby')
        self._port_model(LinearSVC(C=1., random_state=0))

    def tearDown(self):
        super(LinearSVCTest, self).tearDown()
