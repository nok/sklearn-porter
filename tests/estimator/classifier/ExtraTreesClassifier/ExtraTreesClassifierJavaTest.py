# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class ExtraTreesClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierJavaTest, self).setUp()
        self.estimator = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierJavaTest, self).tearDown()

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__digits_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__digits_data__default(self):
        pass
