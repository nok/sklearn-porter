# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.JavaScript import JavaScript


class ExtraTreesClassifierJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierJSTest, self).setUp()
        self.mdl = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierJSTest, self).tearDown()
