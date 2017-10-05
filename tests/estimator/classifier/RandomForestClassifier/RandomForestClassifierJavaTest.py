# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from ..Classifier import Classifier
from ....language.Java import Java


class RandomForestClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierJavaTest, self).setUp()
        self.mdl = RandomForestClassifier(n_estimators=100, random_state=0)

    def tearDown(self):
        super(RandomForestClassifierJavaTest, self).tearDown()
