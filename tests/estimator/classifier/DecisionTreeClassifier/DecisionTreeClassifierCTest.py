# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.C import C


class DecisionTreeClassifierCTest(C, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierCTest, self).setUp()
        self.mdl = DecisionTreeClassifier(random_state=0)

    def tearDown(self):
        super(DecisionTreeClassifierCTest, self).tearDown()
