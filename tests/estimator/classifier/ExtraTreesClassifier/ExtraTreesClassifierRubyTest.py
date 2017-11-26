# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Ruby import Ruby


class ExtraTreesClassifierRubyTest(Ruby, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierRubyTest, self).setUp()
        self.estimator = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierRubyTest, self).tearDown()
