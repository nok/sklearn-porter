# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from ..Classifier import Classifier
from ...language.PHP import PHP


class RandomForestClassifierPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierPHPTest, self).setUp()
        self.mdl = RandomForestClassifier(n_estimators=100, random_state=0)

    def tearDown(self):
        super(RandomForestClassifierPHPTest, self).tearDown()

    @unittest.skip('The generated code would be too large.')
    def test_existing_features_w_digits_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features_w_digits_data(self):
        pass
