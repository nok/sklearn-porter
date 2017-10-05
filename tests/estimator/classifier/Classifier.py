# -*- coding: utf-8 -*-

import os
import numpy as np
import subprocess as subp

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

from ...utils.Timer import Timer


class Classifier(Timer):

    N_RANDOM_FEATURE_SETS = 30
    N_EXISTING_FEATURE_SETS = 30

    def setUp(self):
        np.random.seed(5)
        self._init_env()
        self._start_test()

    def tearDown(self):
        self._clear_model()
        self._stop_test()

    def _init_env(self):
        for param in ['N_RANDOM_FEATURE_SETS', 'N_EXISTING_FEATURE_SETS']:
            n = os.environ.get(param, None)
            if n is not None and str(n).strip().isdigit():
                n = int(n)
                if n > 0:
                    self.__setattr__(param, n)

    def load_binary_data(self, shuffled=True):
        samples = load_breast_cancer()
        self.X = shuffle(samples.data) if shuffled else samples.data
        self.y = shuffle(samples.target) if shuffled else samples.target
        self.n_features = len(self.X[0])

    def load_iris_data(self, shuffled=True):
        samples = load_iris()
        self.X = shuffle(samples.data) if shuffled else samples.data
        self.y = shuffle(samples.target) if shuffled else samples.target
        self.n_features = len(self.X[0])

    def load_digits_data(self, shuffled=True):
        samples = load_digits()
        self.X = shuffle(samples.data) if shuffled else samples.data
        self.y = shuffle(samples.target) if shuffled else samples.target
        self.n_features = len(self.X[0])

    def test_random_features_w_binary_data(self):
        self.load_binary_data()
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

    def test_random_features_w_iris_data(self):
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

    def test_random_features_w_digits_data(self):
        self.load_digits_data()
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

    def test_existing_features_w_binary_data(self):
        self.load_binary_data()
        self._port_model()
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features_w_iris_data(self):
        self.load_iris_data()
        self._port_model()
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features_w_digits_data(self):
        self.load_digits_data()
        self._port_model()
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def _clear_model(self):
        self.mdl = None
        cmd = 'rm -rf tmp'.split()
        subp.call(cmd)
