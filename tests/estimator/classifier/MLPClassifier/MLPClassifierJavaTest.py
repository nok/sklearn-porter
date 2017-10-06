# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class MLPClassifierJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(MLPClassifierJavaTest, self).setUp()
        self.mdl = MLPClassifier(activation='relu',
                                 hidden_layer_sizes=15,
                                 max_iter=500, alpha=1e-4,
                                 solver='sgd', tol=1e-4,
                                 learning_rate_init=.1,
                                 random_state=1,)

    def tearDown(self):
        super(MLPClassifierJavaTest, self).tearDown()

    def test_activation_fn_relu(self):
        self.mdl = MLPClassifier(activation='relu',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu_w_mult_layers(self):
        self.mdl = MLPClassifier(activation='relu',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu_w_binary_data(self):
        self.mdl = MLPClassifier(activation='relu',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_relu_w_mult_layers_w_binary_data(self):
        self.mdl = MLPClassifier(activation='relu',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity(self):
        self.mdl = MLPClassifier(activation='identity',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity_w_mult_layers(self):
        self.mdl = MLPClassifier(activation='identity',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity_w_binary_data(self):
        self.mdl = MLPClassifier(activation='identity',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_identity_w_mult_layers_w_binary_data(self):
        self.mdl = MLPClassifier(activation='identity',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh(self):
        self.mdl = MLPClassifier(activation='tanh',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh_w_mult_layers(self):
        self.mdl = MLPClassifier(activation='tanh',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh_w_binary_data(self):
        self.mdl = MLPClassifier(activation='tanh',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_tanh_w_mult_layers_w_binary_data(self):
        self.mdl = MLPClassifier(activation='tanh',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic(self):
        self.mdl = MLPClassifier(activation='logistic',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic_w_mult_layers(self):
        self.mdl = MLPClassifier(activation='logistic',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_iris_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic_w_binary_data(self):
        self.mdl = MLPClassifier(activation='logistic',
                                 hidden_layer_sizes=15,
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_activation_fn_logistic_w_mult_layers_w_binary_data(self):
        self.mdl = MLPClassifier(activation='logistic',
                                 hidden_layer_sizes=[15, 5],
                                 learning_rate_init=.1)
        self.load_binary_data()
        self._port_model()
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_model()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)
