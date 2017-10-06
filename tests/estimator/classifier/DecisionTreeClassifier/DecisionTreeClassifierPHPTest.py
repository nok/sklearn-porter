# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.PHP import PHP


class DecisionTreeClassifierPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierPHPTest, self).setUp()
        self.mdl = DecisionTreeClassifier(random_state=0)

    def tearDown(self):
        super(DecisionTreeClassifierPHPTest, self).tearDown()
