# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.ensemble import ExtraTreesClassifier

from ..Classifier import Classifier
from ....language.JavaScript import JavaScript as JS


class ExtraTreesClassifierJSTest(JS, Classifier, TestCase):

    def setUp(self):
        super(ExtraTreesClassifierJSTest, self).setUp()
        self.mdl = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierJSTest, self).tearDown()
