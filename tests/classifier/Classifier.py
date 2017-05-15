# -*- coding: utf-8 -*-

import random
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils import shuffle


class Classifier(object):

    def setUp(self):
        samples = load_iris()
        self.X = shuffle(samples.data, random_state=0)
        self.y = shuffle(samples.target, random_state=0)
        self.n_features = len(self.X[0])

    def test_random_features(self):
        self._port_model(self.mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(Y, Y_py)

    def test_existing_features(self):
        self._port_model(self.mdl)
        Y, Y_py = [], []
        for X in self.X:
            Y.append(self.make_pred_in_custom(X))
            Y_py.append(self.make_pred_in_py(X))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(Y, Y_py)