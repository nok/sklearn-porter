# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm import LinearSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.PHP import PHP


class LinearSVCPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCPHPTest, self).setUp()
        self.estimator = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCPHPTest, self).tearDown()
