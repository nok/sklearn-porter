# -*- coding: utf-8 -*-

import unittest

from sklearn.svm.classes import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class NuSVCJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(NuSVCJavaTest, self).setUp()
        self.mdl = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCJavaTest, self).tearDown()

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
