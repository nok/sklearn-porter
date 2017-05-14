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
            activation='identity', hidden_layer_sizes=15,
            max_iter=500, alpha=1e-4, solver='sgd', tol=1e-4,
            random_state=1, learning_rate_init=.1)
        self._port_model(mdl)

    def tearDown(self):
        super(MLPClassifierTest, self).tearDown()

    def test_activation_fn_relu(self):
        mdl = MLPClassifier(
            activation='relu', hidden_layer_sizes=15,
            learning_rate_init=.1)
        self._port_model(mdl)
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(50):
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
        for n in range(50):
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
        for n in range(50):
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
        for n in range(50):
            x = [float(random.uniform(min_vals[f], max_vals[f]))
                 for f in range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        self.assertListEqual(Y, Y_py)

    # def test_activation_fn_tanh(self):
    #     mdl = MLPClassifier(
    #         activation='tanh', hidden_layer_sizes=15,
    #         learning_rate_init=.1)
    #     self._port_model(mdl)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(50):
    #         x = [float(random.uniform(min_vals[f], max_vals[f]))
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #
    #     matches = np.sum(np.array(Y) == np.array(Y_py))
    #     accuracy = matches / 50. * 100.
    #     print("accuracy: %s" % repr(accuracy))
    #
    #     self.assertListEqual(Y, Y_py)
    #
    # def test_activation_fn_tanh_with_mult_layers(self):
    #     mdl = MLPClassifier(
    #         activation='tanh', hidden_layer_sizes=[15, 5],
    #         learning_rate_init=.1)
    #     self._port_model(mdl)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(50):
    #         x = [float(random.uniform(min_vals[f], max_vals[f]))
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #
    #     matches = np.sum(np.array(Y) == np.array(Y_py))
    #     accuracy = matches / 50. * 100.
    #     print("accuracy: %s" % repr(accuracy))
    #
    #     self.assertListEqual(Y, Y_py)

    # def test_activation_fn_logistic(self):
    #     mdl = MLPClassifier(
    #         activation='logistic', hidden_layer_sizes=15,
    #         learning_rate_init=.1)
    #     self._port_model(mdl)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(30):
    #         x = [float(random.uniform(min_vals[f], max_vals[f]))
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #     self.assertListEqual(Y, Y_py)
    #
    # def test_activation_fn_logistic_with_mult_layers(self):
    #     mdl = MLPClassifier(
    #         activation='logistic', hidden_layer_sizes=[15, 10],
    #         learning_rate_init=.1)
    #     self._port_model(mdl)
    #     Y, Y_py = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(30):
    #         x = [float(random.uniform(min_vals[f], max_vals[f]))
    #              for f in range(self.n_features)]
    #         Y.append(self.make_pred_in_custom(x))
    #         Y_py.append(self.make_pred_in_py(x))
    #     self.assertListEqual(Y, Y_py)
