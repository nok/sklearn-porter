# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.PHP import PHP


class ExtraTreesClassifierPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierPHPTest, self).setUp()
        self.estimator = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierPHPTest, self).tearDown()

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__digits_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__digits_data__default(self):
        pass
