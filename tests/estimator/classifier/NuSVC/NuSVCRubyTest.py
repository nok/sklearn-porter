# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.svm.classes import NuSVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Ruby import Ruby


class NuSVCRubyTest(Ruby, Classifier, TestCase):

    def setUp(self):
        super(NuSVCRubyTest, self).setUp()
        self.mdl = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCRubyTest, self).tearDown()
