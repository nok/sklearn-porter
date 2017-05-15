# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ...language.PHP import PHP


class DecisionTreeClassifierPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierPHPTest, self).setUp()
        mdl = DecisionTreeClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(DecisionTreeClassifierPHPTest, self).tearDown()
