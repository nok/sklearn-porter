# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.C import C


class ExtraTreesClassifierCTest(C, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierCTest, self).setUp()
        self.mdl = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierCTest, self).tearDown()
