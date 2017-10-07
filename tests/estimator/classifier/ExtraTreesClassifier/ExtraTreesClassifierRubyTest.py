# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import ExtraTreesClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Ruby import Ruby


class ExtraTreesClassifierRubyTest(Ruby, Classifier, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierRubyTest, self).setUp()
        self.estimator = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierRubyTest, self).tearDown()
