# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class RandomForestClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierJavaTest, self).setUp()
        mdl = RandomForestClassifier(n_estimators=100, random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(RandomForestClassifierJavaTest, self).tearDown()
