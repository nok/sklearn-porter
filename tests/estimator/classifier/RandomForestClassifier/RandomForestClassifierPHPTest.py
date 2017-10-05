# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier

from ..Classifier import Classifier
from ....language.PHP import PHP


class RandomForestClassifierPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(RandomForestClassifierPHPTest, self).setUp()
        self.mdl = RandomForestClassifier(n_estimators=20, random_state=0)

    def tearDown(self):
        super(RandomForestClassifierPHPTest, self).tearDown()
