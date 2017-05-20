# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from ..Classifier import Classifier
from ...language.C import C


class LinearSVCCTest(C, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCCTest, self).setUp()
        self.load_multiclass_data()
        mdl = LinearSVC(C=1., random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(LinearSVCCTest, self).tearDown()
