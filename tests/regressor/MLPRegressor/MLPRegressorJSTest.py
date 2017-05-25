# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import random

from sklearn.neural_network.multilayer_perceptron import MLPRegressor

from ..Regressor import Regressor
from ...language.JavaScript import JavaScript as JS


class MLPRegressorJSTest(JS, Regressor, TestCase):

    N_RANDOM_TESTS = 50

    def setUp(self):
        super(MLPRegressorJSTest, self).setUp()
        np.random.seed(0)
        self.mdl = MLPRegressor(activation='relu', hidden_layer_sizes=50,
                                max_iter=500, learning_rate_init=.1,
                                random_state=3)

    def tearDown(self):
        super(MLPRegressorJSTest, self).tearDown()

    def test_activation_fn_relu_with_mult_layers_2(self):
        self.mdl = MLPRegressor(activation='relu',
                                hidden_layer_sizes=(50, 30),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_relu_with_mult_layers_3(self):
        self.mdl = MLPRegressor(activation='relu',
                                hidden_layer_sizes=(50, 30, 15),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity(self):
        self.mdl = MLPRegressor(activation='identity',
                                hidden_layer_sizes=50,
                                max_iter=500,
                                learning_rate_init=.1)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_2(self):
        self.mdl = MLPRegressor(activation='identity',
                                hidden_layer_sizes=(50, 30),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_3(self):
        self.mdl = MLPRegressor(activation='identity',
                                hidden_layer_sizes=(50, 30, 15),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh(self):
        self.mdl = MLPRegressor(activation='tanh',
                                hidden_layer_sizes=50,
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_2(self):
        self.mdl = MLPRegressor(activation='tanh',
                                hidden_layer_sizes=(50, 30),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_3(self):
        self.mdl = MLPRegressor(activation='tanh',
                                hidden_layer_sizes=(50, 30, 15),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logistic(self):
        self.mdl = MLPRegressor(activation='logistic',
                                hidden_layer_sizes=50,
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_2(self):
        self.mdl = MLPRegressor(activation='logistic',
                                hidden_layer_sizes=(50, 30),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_3(self):
        self.mdl = MLPRegressor(activation='logistic',
                                hidden_layer_sizes=(50, 30, 15),
                                max_iter=500,
                                learning_rate_init=.1,
                                random_state=3)
        self.load_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))
