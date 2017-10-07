# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.naive_bayes import GaussianNB

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class GaussianNBJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(GaussianNBJavaTest, self).setUp()
        self.estimator = GaussianNB()

    def tearDown(self):
        super(GaussianNBJavaTest, self).tearDown()
