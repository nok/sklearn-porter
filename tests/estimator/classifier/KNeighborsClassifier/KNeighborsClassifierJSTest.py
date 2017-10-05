# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier

from ..Classifier import Classifier
from ....language.JavaScript import JavaScript as JS


class KNeighborsClassifierJSTest(JS, Classifier, TestCase):

    def setUp(self):
        super(KNeighborsClassifierJSTest, self).setUp()
        self.mdl = KNeighborsClassifier(n_neighbors=3)

    def tearDown(self):
        super(KNeighborsClassifierJSTest, self).tearDown()
