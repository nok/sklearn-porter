# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ...language.C import C


class DecisionTreeClassifierCTest(C, Classifier, TestCase):

    def setUp(self):
        super(DecisionTreeClassifierCTest, self).setUp()
        mdl = DecisionTreeClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(DecisionTreeClassifierCTest, self).tearDown()
