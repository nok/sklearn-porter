# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class ExtraTreesClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierJavaTest, self).setUp()
        mdl = ExtraTreesClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(ExtraTreesClassifierJavaTest, self).tearDown()
