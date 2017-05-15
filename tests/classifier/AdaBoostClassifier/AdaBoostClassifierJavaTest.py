# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class AdaBoostClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(AdaBoostClassifierJavaTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        mdl = AdaBoostClassifier(base_estimator=base_estimator,
                                 n_estimators=100,
                                 random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(AdaBoostClassifierJavaTest, self).tearDown()
