# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import LinearSVC

from ..Classifier import Classifier
from ...language.Ruby import Ruby


class LinearSVCRubyTest(Ruby, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCRubyTest, self).setUp()
        mdl = LinearSVC(C=1., random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(LinearSVCRubyTest, self).tearDown()
