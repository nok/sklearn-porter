import unittest
from ..JavaTest import JavaTest
import numpy as np
import random

from sklearn.svm.classes import SVC
from sklearn_porter import Porter


class SVCTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(SVCTest, self).setUp()
        self.porter = Porter(language='java')
        clf = SVC(C=1., kernel='rbf', gamma=0.001, random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(SVCTest, self).tearDown()

    def test_kernel_linear(self):
        clf = SVC(C=1., kernel='linear', gamma=0.001, random_state=0)
        self.set_classifier(clf)
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            java_preds.append(self.make_pred_in_java(x))
            py_preds.append(self.make_pred_in_py(x))
        self.assertListEqual(py_preds, java_preds)

    def test_kernel_poly(self):
        clf = SVC(C=1., kernel='poly', gamma=0.001, random_state=0)
        self.set_classifier(clf)
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            java_preds.append(self.make_pred_in_java(x))
            py_preds.append(self.make_pred_in_py(x))
        self.assertListEqual(py_preds, java_preds)

    def test_kernel_sigmoid(self):
        clf = SVC(C=1., kernel='sigmoid', gamma=0.001, random_state=0)
        self.set_classifier(clf)
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            java_preds.append(self.make_pred_in_java(x))
            py_preds.append(self.make_pred_in_py(x))
        self.assertListEqual(py_preds, java_preds)
