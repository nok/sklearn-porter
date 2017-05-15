# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from ..Classifier import Classifier
from ...language.C import C


class ExtraTreesClassifierCTest(C, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierCTest, self).setUp()
        mdl = ExtraTreesClassifier(random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(ExtraTreesClassifierCTest, self).tearDown()
