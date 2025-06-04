# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.PHP import PHP


class NuSVCPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(NuSVCPHPTest, self).setUp()
        self.estimator = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCPHPTest, self).tearDown()
