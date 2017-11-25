# -*- coding: utf-8 -*-

import unittest
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.ClassifierExported import ClassifierExported as Classifier
from tests.language.Java import Java


class AdaBoostClassifierJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(AdaBoostClassifierJavaTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        self.estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                            n_estimators=100, random_state=0)

    def tearDown(self):
        super(AdaBoostClassifierJavaTest, self).tearDown()
