# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import random

from sklearn.svm.classes import SVC

from ..Classifier import Classifier
from ...language.JavaScript import JavaScript as JS


class SVCJSTest(JS, Classifier, TestCase):

    def setUp(self):
        super(SVCJSTest, self).setUp()
        self.mdl = SVC(C=1., kernel='rbf',
                       gamma=0.001,
                       random_state=0)

    def tearDown(self):
        super(SVCJSTest, self).tearDown()

    def test_linear_kernel(self):
        self.mdl = SVC(C=1., kernel='linear',
                       gamma=0.001,
                       random_state=0)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_sigmoid_kernel(self):
        self.mdl = SVC(C=1., kernel='sigmoid',
                       gamma=0.001,
                       random_state=0)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)
