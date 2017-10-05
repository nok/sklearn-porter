# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ....language.JavaScript import JavaScript as JS


class DecisionTreeClassifierJSTest(JS, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierJSTest, self).setUp()
        self.mdl = DecisionTreeClassifier(random_state=0)

    def tearDown(self):
        super(DecisionTreeClassifierJSTest, self).tearDown()
