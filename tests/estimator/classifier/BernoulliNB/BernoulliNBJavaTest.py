# -*- coding: utf-8 -*-

import unittest

from sklearn.naive_bayes import BernoulliNB

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class BernoulliNBJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(BernoulliNBJavaTest, self).setUp()
        self.mdl = BernoulliNB()

    def tearDown(self):
        super(BernoulliNBJavaTest, self).tearDown()

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features_w_iris_data(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features_w_binary_data(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features_w_binary_data(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features_w_digits_data(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features_w_digits_data(self):
        pass