# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class DecisionTreeClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierJavaTest, self).setUp()
        mdl = DecisionTreeClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(DecisionTreeClassifierJavaTest, self).tearDown()
