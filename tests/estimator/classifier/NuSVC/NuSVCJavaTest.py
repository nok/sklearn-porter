# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.svm.classes import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.Java import Java


class NuSVCJavaTest(Java, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(NuSVCJavaTest, self).setUp()
        self.estimator = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCJavaTest, self).tearDown()

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
