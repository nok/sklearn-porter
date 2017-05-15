# -*- coding: utf-8 -*-

import unittest

from sklearn.naive_bayes import BernoulliNB

from ..Classifier import Classifier
from ...language.Java import Java


class BernoulliNBJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(BernoulliNBJavaTest, self).setUp()
        mdl = BernoulliNB()
        self._port_model(mdl)

    def tearDown(self):
        super(BernoulliNBJavaTest, self).tearDown()

    @unittest.skip('BernoulliNB is suitable for discrete data.')
    def test_random_features(self):
        pass