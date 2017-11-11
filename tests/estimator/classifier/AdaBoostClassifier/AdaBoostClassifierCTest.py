# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.C import C


class AdaBoostClassifierCTest(C, Classifier, unittest.TestCase):

    def setUp(self):
        super(AdaBoostClassifierCTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        self.estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                            n_estimators=100, random_state=0)

    def tearDown(self):
        super(AdaBoostClassifierCTest, self).tearDown()
