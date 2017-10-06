# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.JavaScript import JavaScript


class LinearSVCJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCJSTest, self).setUp()
        self.mdl = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCJSTest, self).tearDown()
