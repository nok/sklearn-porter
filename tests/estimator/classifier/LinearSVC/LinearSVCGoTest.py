# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Go import Go


class LinearSVCGoTest(Go, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCGoTest, self).setUp()
        self.mdl = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCGoTest, self).tearDown()
