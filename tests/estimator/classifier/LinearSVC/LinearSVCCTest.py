# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.C import C


class LinearSVCCTest(C, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCCTest, self).setUp()
        self.estimator = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCCTest, self).tearDown()
