# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Go import Go


class RandomForestClassifierGoTest(Go, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierGoTest, self).setUp()
        self.estimator = RandomForestClassifier(n_estimators=100,
                                                random_state=0)

    def tearDown(self):
        super(RandomForestClassifierGoTest, self).tearDown()
