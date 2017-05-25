# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.naive_bayes import GaussianNB

from ..Classifier import Classifier
from ...language.Java import Java


class GaussianNBJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(GaussianNBJavaTest, self).setUp()
        self.mdl = GaussianNB()

    def tearDown(self):
        super(GaussianNBJavaTest, self).tearDown()
