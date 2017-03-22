# -*- coding: utf-8 -*-

import unittest

from sklearn.naive_bayes import BernoulliNB
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class BernoulliNBTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(BernoulliNBTest, self).setUp()
        mdl = BernoulliNB()
        self._port_model(mdl)

    def tearDown(self):
        super(BernoulliNBTest, self).tearDown()

    @unittest.skip('BernoulliNB is suitable for discrete data.')
    def test_random_features(self):
        pass
