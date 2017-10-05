# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from ..Classifier import Classifier
from ....language.Ruby import Ruby


class RandomForestClassifierRubyTest(Ruby, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierRubyTest, self).setUp()
        self.mdl = RandomForestClassifier(n_estimators=20, random_state=0)

    def tearDown(self):
        super(RandomForestClassifierRubyTest, self).tearDown()
