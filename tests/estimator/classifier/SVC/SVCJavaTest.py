# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import numpy as np

from sklearn.svm.classes import SVC

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.Java import Java


class SVCJavaTest(Java, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(SVCJavaTest, self).setUp()
        self.estimator = SVC(C=1., kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(SVCJavaTest, self).tearDown()

    def test_linear_kernel(self):
        self.estimator = SVC(C=1., kernel='linear',
                             gamma=0.001, random_state=0)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_sigmoid_kernel(self):
        self.estimator = SVC(C=1., kernel='sigmoid',
                             gamma=0.001, random_state=0)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__binary_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__binary_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_existing_features__digits_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_random_features__digits_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_rbf_kernel__binary_data__default(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_linear_kernel_w_binary_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_poly_kernel(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_poly_kernel_w_binary_data(self):
        pass

    @unittest.skip('The generated code would be too large.')
    def test_sigmoid_kernel_w_binary_data(self):
        pass
