# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.JavaScript import JavaScript


class AdaBoostClassifierJSTest(JavaScript, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(AdaBoostClassifierJSTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        self.estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                            n_estimators=100, random_state=0)

    def tearDown(self):
        super(AdaBoostClassifierJSTest, self).tearDown()