# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.JavaScript import JavaScript


class DecisionTreeClassifierJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierJSTest, self).setUp()
        self.mdl = DecisionTreeClassifier(random_state=0)

    def tearDown(self):
        super(DecisionTreeClassifierJSTest, self).tearDown()
