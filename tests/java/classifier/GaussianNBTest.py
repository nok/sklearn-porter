# -*- coding: utf-8 -*-

import unittest

from sklearn.naive_bayes import GaussianNB
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class GaussianNBTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(GaussianNBTest, self).setUp()
        mdl = GaussianNB()
        self._port_model(mdl)

    def tearDown(self):
        super(GaussianNBTest, self).tearDown()
