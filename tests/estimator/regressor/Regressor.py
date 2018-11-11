# -*- coding: utf-8 -*-

import os
import subprocess as subp

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


class Regressor(object):

    TEST_N_RANDOM_FEATURE_SETS = 20
    TEST_N_EXISTING_FEATURE_SETS = 20

    def setUp(self):
        np.random.seed(5)
        self._init_env()
        self.load_data()

    def tearDown(self):
        self._clear_estimator()

    def _init_env(self):
        for param in ['TEST_N_RANDOM_FEATURE_SETS', 'TEST_N_EXISTING_FEATURE_SETS']:
            n = os.environ.get(param, None)
            if n is not None and str(n).strip().isdigit():
                n = int(n)
                if n > 0:
                    self.__setattr__(param, n)

    def load_data(self, shuffled=True):
        samples = load_diabetes()
        self.X = shuffle(samples.data) if shuffled else samples.data
        self.y = shuffle(samples.target) if shuffled else samples.target
        self.n_features = len(self.X[0])

    def test_random_features_new(self):
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_existing_features_new(self):
        self.load_data()
        self._port_estimator()
        match = []
        n = min(self.TEST_N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def _clear_estimator(self):
        self.estimator = None
        subp.call('rm -rf tmp'.split())
