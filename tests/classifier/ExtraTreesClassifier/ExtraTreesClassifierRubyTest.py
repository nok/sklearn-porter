# -*- coding: utf-8 -*-

import unittest

from sklearn.ensemble import ExtraTreesClassifier

from ..Classifier import Classifier
from ...language.Ruby import Ruby


class ExtraTreesClassifierRubyTest(Ruby, Classifier, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierRubyTest, self).setUp()
        self.mdl = ExtraTreesClassifier(random_state=0)

    def tearDown(self):
        super(ExtraTreesClassifierRubyTest, self).tearDown()

    # @unittest.skip('The generated code would be too large.')
    # def test_existing_features_w_digits_data(self):
    #     pass
    #
    # @unittest.skip('The generated code would be too large.')
    # def test_random_features_w_digits_data(self):
    #     pass
