# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.naive_bayes import BernoulliNB

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.Java import Java


class BernoulliNBJavaTest(Java, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(BernoulliNBJavaTest, self).setUp()
        self.estimator = BernoulliNB()

    def tearDown(self):
        super(BernoulliNBJavaTest, self).tearDown()

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__iris_data__default(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features__binary_data__default(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__binary_data__default(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__digits_data__default(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features__digits_data__default(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__iris_data__exported(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features__binary_data__exported(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__binary_data__exported(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_random_features__digits_data__exported(self):
        pass

    @unittest.skip('BernoulliNB is just suitable for discrete data.')
    def test_existing_features__digits_data__exported(self):
        pass