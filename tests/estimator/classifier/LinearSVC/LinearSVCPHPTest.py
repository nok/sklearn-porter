# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from ..Classifier import Classifier
from ....language.PHP import PHP


class LinearSVCPHPTest(PHP, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCPHPTest, self).setUp()
        self.mdl = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCPHPTest, self).tearDown()
