# -*- coding: utf-8 -*-

import os
import subprocess as subp

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

from ..utils.Timer import Timer


class Regressor(Timer):

    N_RANDOM_FEATURE_SETS = 30
    N_EXISTING_FEATURE_SETS = 30

    def setUp(self):
        np.random.seed(5)
        self._init_env()
        self._start_test()
        self.load_data()

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

    def load_data(self, shuffled=True):
        samples = load_diabetes()
        self.X = shuffle(samples.data) if shuffled else samples.data
        self.y = shuffle(samples.target) if shuffled else samples.target
        self.n_features = len(self.X[0])

    def test_random_features_new(self):
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_existing_features_new(self):
        self._port_model()
        match = []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def _clear_model(self):
        self.mdl = None
        subp.call(['rm', '-rf', 'tmp'])