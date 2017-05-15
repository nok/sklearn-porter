# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import random

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from ..Classifier import Classifier
from ...language.Java import Java


class MLPClassifierJavaTest(Java, Classifier, TestCase):

    N_RANDOM_TESTS = 60

    def setUp(self):
        super(MLPClassifierJavaTest, self).setUp()
        mdl = MLPClassifier(activation='identity',
                            hidden_layer_sizes=15,
                            max_iter=500, alpha=1e-4,
                            solver='sgd', tol=1e-4,
                            random_state=1,
                            learning_rate_init=.1)
        self._port_model(mdl)

    def tearDown(self):
        super(MLPClassifierJavaTest, self).tearDown()

    def test_activation_fn_relu(self):
        mdl = MLPClassifier(
            activation='relu', hidden_layer_sizes=15,
            learning_rate_init=.1)
        self._port_model(mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [float(random.uniform(min_vals[f], max_vals[f]))
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_relu_with_mult_layers(self):
        mdl = MLPClassifier(
            activation='relu', hidden_layer_sizes=[15, 5],
            learning_rate_init=.1)
        self._port_model(mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [float(random.uniform(min_vals[f], max_vals[f]))
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_identity(self):
        mdl = MLPClassifier(
            activation='identity', hidden_layer_sizes=15,
            learning_rate_init=.1)
        self._port_model(mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [float(random.uniform(min_vals[f], max_vals[f]))
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_identity_with_mult_layers(self):
        mdl = MLPClassifier(
            activation='identity', hidden_layer_sizes=[15, 5],
            learning_rate_init=.1)
        self._port_model(mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [float(random.uniform(min_vals[f], max_vals[f]))
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)
