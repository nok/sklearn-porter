# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from sklearn.svm import SVC

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Ruby import Ruby


class SVCRubyTest(Ruby, Classifier, TestCase):

    def setUp(self):
        super(SVCRubyTest, self).setUp()
        self.estimator = SVC(C=1., kernel='rbf',
                             gamma=0.001, random_state=0)

    def tearDown(self):
        super(SVCRubyTest, self).tearDown()

    def test_linear_kernel(self):
        self.estimator = SVC(C=1., kernel='linear',
                             gamma=0.001, random_state=0)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
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
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_auto_gamma(self):
        self.estimator = SVC(C=1., gamma='auto', random_state=0)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)
