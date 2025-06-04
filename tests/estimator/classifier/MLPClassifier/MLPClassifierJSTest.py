# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from sklearn.neural_network import MLPClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.estimator.classifier.ExportedData import ExportedData
from tests.language.JavaScript import JavaScript


class MLPClassifierJSTest(JavaScript, Classifier, ExportedData, TestCase):

    def setUp(self):
        super(MLPClassifierJSTest, self).setUp()
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=15,
                                       max_iter=500, alpha=1e-4,
                                       solver='sgd', tol=1e-4,
                                       learning_rate_init=.1,
                                       random_state=1, )

    def tearDown(self):
        super(MLPClassifierJSTest, self).tearDown()

    def test_activation_fn_relu(self):
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu__mult_layers(self):
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu__binary_data(self):
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu__mult_layers__binary_data(self):
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity(self):
        self.estimator = MLPClassifier(activation='identity',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity__mult_layers(self):
        self.estimator = MLPClassifier(activation='identity',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity__binary_data(self):
        self.estimator = MLPClassifier(activation='identity',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity__mult_layers__binary_data(self):
        self.estimator = MLPClassifier(activation='identity',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh(self):
        self.estimator = MLPClassifier(activation='tanh',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh__mult_layers(self):
        self.estimator = MLPClassifier(activation='tanh',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh__binary_data(self):
        self.estimator = MLPClassifier(activation='tanh',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh__mult_layers__binary_data(self):
        self.estimator = MLPClassifier(activation='tanh',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic(self):
        self.estimator = MLPClassifier(activation='logistic',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic__mult_layers(self):
        self.estimator = MLPClassifier(activation='logistic',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_iris_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic__binary_data(self):
        self.estimator = MLPClassifier(activation='logistic',
                                       hidden_layer_sizes=15,
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic__mult_layers__binary_data(self):
        self.estimator = MLPClassifier(activation='logistic',
                                       hidden_layer_sizes=[15, 5],
                                       learning_rate_init=.1)
        self.load_binary_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu__multi_layers__image_data(self):
        self.estimator = MLPClassifier(activation='relu',
                                       hidden_layer_sizes=[15, 10, 5],
                                       learning_rate_init=.1)
        self.load_digits_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh__multi_layers__image_data(self):
        self.estimator = MLPClassifier(activation='tanh',
                                       hidden_layer_sizes=[15, 10, 5],
                                       learning_rate_init=.1)
        self.load_digits_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic__multi_layers__image_data(self):
        self.estimator = MLPClassifier(activation='logistic',
                                       hidden_layer_sizes=[15, 10, 5],
                                       learning_rate_init=.1)
        self.load_digits_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity__multi_layers__image_data(self):
        self.estimator = MLPClassifier(activation='identity',
                                       hidden_layer_sizes=[15, 10, 5],
                                       learning_rate_init=.1)
        self.load_digits_data()
        self._port_estimator()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.TEST_N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)
