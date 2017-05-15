# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.neighbors import KNeighborsClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class KNeighborsClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(KNeighborsClassifierJavaTest, self).setUp()
        mdl = KNeighborsClassifier(algorithm='brute',
                                   n_neighbors=3,
                                   weights='uniform')
        self._port_model(mdl)

    def tearDown(self):
        super(KNeighborsClassifierJavaTest, self).tearDown()
