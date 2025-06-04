# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.C import C


class NuSVCCTest(C, Classifier, TestCase):

    def setUp(self):
        super(NuSVCCTest, self).setUp()
        self.estimator = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCCTest, self).tearDown()
