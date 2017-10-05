# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ....language.Java import Java


class AdaBoostClassifierJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(AdaBoostClassifierJavaTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        self.mdl = AdaBoostClassifier(base_estimator=base_estimator,
                                      n_estimators=100, random_state=0)

    def tearDown(self):
        super(AdaBoostClassifierJavaTest, self).tearDown()

    @unittest.skip('Skip random features test.')
    def test_random_features_w_iris_data(self):
        pass

    @unittest.skip('Skip random features test.')
    def test_random_features_w_binary_data(self):
        pass

    @unittest.skip('TODO')
    def test_random_features_w_digits_data(self):
        pass

    @unittest.skip('TODO')
    def test_existing_features_w_digits_data(self):
        pass