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
        mdl = MLPRegressor(activation='relu',
                           hidden_layer_sizes=50,
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)

    def tearDown(self):
        super(MLPRegressorJSTest, self).tearDown()

    def test_activation_fn_relu_with_mult_layers_2(self):
        mdl = MLPRegressor(activation='relu',
                           hidden_layer_sizes=(50, 30),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_relu_with_mult_layers_3(self):
        mdl = MLPRegressor(activation='relu',
                           hidden_layer_sizes=(50, 30, 15),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity(self):
        mdl = MLPRegressor(activation='identity',
                           hidden_layer_sizes=50,
                           max_iter=500,
                           learning_rate_init=.1)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_2(self):
        mdl = MLPRegressor(activation='identity',
                           hidden_layer_sizes=(50, 30),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_identity_with_mult_layers_3(self):
        mdl = MLPRegressor(activation='identity',
                           hidden_layer_sizes=(50, 30, 15),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh(self):
        mdl = MLPRegressor(activation='tanh',
                           hidden_layer_sizes=50,
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_2(self):
        mdl = MLPRegressor(activation='tanh',
                           hidden_layer_sizes=(50, 30),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_tanh_with_mult_layers_3(self):
        mdl = MLPRegressor(activation='tanh',
                           hidden_layer_sizes=(50, 30, 15),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logistic(self):
        mdl = MLPRegressor(activation='logistic',
                           hidden_layer_sizes=50,
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_2(self):
        mdl = MLPRegressor(activation='logistic',
                           hidden_layer_sizes=(50, 30),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))

    def test_activation_fn_logstic_with_mult_layers_3(self):
        mdl = MLPRegressor(activation='logistic',
                           hidden_layer_sizes=(50, 30, 15),
                           max_iter=500,
                           learning_rate_init=.1,
                           random_state=3)
        self._port_model(mdl)
        match = []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            match.append(self.make_pred_in_custom(x, cast=False) -
                         self.make_pred_in_py(x, cast=False) < 0.0001)
        # noinspection PyUnresolvedReferences
        self.assertEqual(match.count(True), len(match))
