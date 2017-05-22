# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.naive_bayes import GaussianNB

from ..Classifier import Classifier
from ...language.JavaScript import JavaScript


class GaussianNBJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(GaussianNBJSTest, self).setUp()
        mdl = GaussianNB()
        self._port_model(mdl)

    def tearDown(self):
        super(GaussianNBJSTest, self).tearDown()
