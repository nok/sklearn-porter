# -*- coding: utf-8 -*-

import random
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


class Regressor(object):

    N_RANDOM_TESTS = 5

    def setUp(self):
        samples = load_diabetes()
        self.X = shuffle(samples.data, random_state=0)
        self.y = shuffle(samples.target, random_state=0)
        self.n_features = len(self.X[0])

    def test_random_features(self):
        self._port_model(self.mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            # print(x)
            # print(self.make_pred_in_py(x, cast=False),
            #       self.make_pred_in_custom(x, cast=False))
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_existing_features(self):
        self._port_model(self.mdl)
        match = []
        for X in self.X[-self.N_RANDOM_TESTS:]:
            # print(X)
            # print(self.make_pred_in_py(X, cast=False),
            #       self.make_pred_in_custom(X, cast=False))
            match.append(self.make_pred_in_custom(X, cast=False) -
                         self.make_pred_in_py(X, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))
