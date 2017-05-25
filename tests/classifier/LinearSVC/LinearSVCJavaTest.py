# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from ..Classifier import Classifier
from ...language.Java import Java


class LinearSVCJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCJavaTest, self).setUp()
        self.mdl = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCJavaTest, self).tearDown()
