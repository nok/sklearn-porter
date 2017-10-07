# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class RandomForestClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierJavaTest, self).setUp()
        self.estimator = RandomForestClassifier(n_estimators=100,
                                                random_state=0)

    def tearDown(self):
        super(RandomForestClassifierJavaTest, self).tearDown()
