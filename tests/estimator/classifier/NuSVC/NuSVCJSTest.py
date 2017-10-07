# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.JavaScript import JavaScript


class NuSVCJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(NuSVCJSTest, self).setUp()
        self.estimator = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCJSTest, self).tearDown()
