# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np

from sklearn.neural_network import MLPRegressor

from tests.estimator.regressor.Regressor import Regressor
from tests.language.JavaScript import JavaScript


class MLPRegressorJSTest(JavaScript, Regressor, TestCase):

    N_RANDOM_TESTS = 50

    def setUp(self):
        super(MLPRegressorJSTest, self).setUp()
        np.random.seed(0)
        self.estimator = MLPRegressor(activation='relu', hidden_layer_sizes=50,
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)

    def tearDown(self):
        super(MLPRegressorJSTest, self).tearDown()

    def test_activation_fn_relu_with_mult_layers_2(self):
        self.estimator = MLPRegressor(activation='relu',
                                      hidden_layer_sizes=(50, 30),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_relu_with_mult_layers_3(self):
        self.estimator = MLPRegressor(activation='relu',
                                      hidden_layer_sizes=(50, 30, 15),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity(self):
        self.estimator = MLPRegressor(activation='identity',
                                      hidden_layer_sizes=50, max_iter=500,
                                      learning_rate_init=.1)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_2(self):
        self.estimator = MLPRegressor(activation='identity',
                                      hidden_layer_sizes=(50, 30), max_iter=500,
                                      learning_rate_init=.1, random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_3(self):
        self.estimator = MLPRegressor(activation='identity',
                                      hidden_layer_sizes=(50, 30, 15),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh(self):
        self.estimator = MLPRegressor(activation='tanh', hidden_layer_sizes=50,
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_2(self):
        self.estimator = MLPRegressor(activation='tanh',
                                      hidden_layer_sizes=(50, 30),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_3(self):
        self.estimator = MLPRegressor(activation='tanh',
                                      hidden_layer_sizes=(50, 30, 15),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logistic(self):
        self.estimator = MLPRegressor(activation='logistic',
                                      hidden_layer_sizes=50, max_iter=500,
                                      learning_rate_init=.1, random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_2(self):
        self.estimator = MLPRegressor(activation='logistic',
                                      hidden_layer_sizes=(50, 30), max_iter=500,
                                      learning_rate_init=.1, random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_3(self):
        self.estimator = MLPRegressor(activation='logistic',
                                      hidden_layer_sizes=(50, 30, 15),
                                      max_iter=500, learning_rate_init=.1,
                                      random_state=3)
        self.load_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        match = []
        for _ in range(30):
            x = np.random.uniform(amin, amax, self.n_features)
            match.append(self.pred_in_custom(x, cast=False) -
                         self.pred_in_py(x, cast=False) < 0.0001)
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))
