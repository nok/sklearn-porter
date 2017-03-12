# -*- coding: utf-8 -*-

import unittest
import numpy as np
import random

from sklearn.svm.classes import NuSVC
from sklearn_porter import Porter

from ..JavaScriptTest import JavaScriptTest


class SVCTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(SVCTest, self).setUp()
        mdl = NuSVC(kernel='rbf', gamma=0.001, random_state=0)
        self._port_model(mdl)

    def tearDown(self):
        super(SVCTest, self).tearDown()

    def test_kernel_linear(self):
        clf = NuSVC(kernel='linear', gamma=0.001, random_state=0)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.n_random_tests):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_kernel_poly(self):
        clf = NuSVC(kernel='poly', gamma=0.001, random_state=0)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.n_random_tests):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_kernel_sigmoid(self):
        clf = NuSVC(kernel='sigmoid', gamma=0.001, random_state=0)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.n_random_tests):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)
