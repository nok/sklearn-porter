# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.Java import Java


class KNeighborsClassifierJavaTest(Java, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(KNeighborsClassifierJavaTest, self).setUp()
        self.estimator = KNeighborsClassifier(n_neighbors=3)

    def tearDown(self):
        super(KNeighborsClassifierJavaTest, self).tearDown()

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__binary_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__binary_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__digits_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__digits_data__default(self):
        pass
