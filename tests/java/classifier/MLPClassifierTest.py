# -*- coding: utf-8 -*-

import unittest
import numpy as np
import random

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn_porter import Porter

from ..JavaTest import JavaTest


class MLPClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(MLPClassifierTest, self).setUp()
        mdl = MLPClassifier(
            activation='identity', hidden_layer_sizes=50,
            max_iter=500, alpha=1e-4, solver='sgd', tol=1e-4,
            random_state=1, learning_rate_init=.1)
        self._port_model(mdl)

    def tearDown(self):
        super(MLPClassifierTest, self).tearDown()

    def test_activation_fn_relu(self):
        clf = MLPClassifier(
            activation='relu', hidden_layer_sizes=50,
            learning_rate_init=.1)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_relu_with_mult_layers(self):
        clf = MLPClassifier(
            activation='relu', hidden_layer_sizes=[60, 40, 20],
            learning_rate_init=.1)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_identity(self):
        clf = MLPClassifier(
            activation='identity', hidden_layer_sizes=50,
            learning_rate_init=.1)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    def test_activation_fn_identity_with_mult_layers(self):
        clf = MLPClassifier(
            activation='identity', hidden_layer_sizes=[60, 40, 20],
            learning_rate_init=.1)
        self._port_model(clf)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    # def test_activation_fn_tanh(self):
    #     clf = MLPClassifier(
    #         activation='tanh', hidden_layer_sizes=50,
    #         learning_rate_init=.1)
    #     self._port_model(clf)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(30):
    #         x = [random.uniform(min_vals[f], max_vals[f])
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #     self.assertListEqual(Y, Y_py)

    # def test_activation_fn_logistic(self):
    #     clf = MLPClassifier(
    #         activation='logistic', hidden_layer_sizes=50,
    #         learning_rate_init=.1)
    #     self._port_model(clf)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(30):
    #         x = [random.uniform(min_vals[f], max_vals[f])
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #     self.assertListEqual(Y, Y_py)
