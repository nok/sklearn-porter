# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Go import Go


class DecisionTreeClassifierGoTest(Go, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierGoTest, self).setUp()
        self.mdl = DecisionTreeClassifier(random_state=0)

    def tearDown(self):
        super(DecisionTreeClassifierGoTest, self).tearDown()
