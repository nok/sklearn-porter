# -*- coding: utf-8 -*-

import unittest
from sklearn.neighbors import KNeighborsClassifier

from tests.estimator.classifier.ClassifierExported import ClassifierExported as Classifier
from tests.language.Java import Java


class KNeighborsClassifierJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(KNeighborsClassifierJavaTest, self).setUp()
        self.estimator = KNeighborsClassifier(n_neighbors=3)

    def tearDown(self):
        super(KNeighborsClassifierJavaTest, self).tearDown()

    @unittest.skip('The generated code would be too large.')
    def test_existing_features_w_binary_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features_w_binary_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_existing_features_w_digits_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features_w_digits_data(self):
        pass
