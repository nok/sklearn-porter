import unittest
from ..JavaScriptTest import JavaScriptTest
import numpy as np
import random

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn_porter import Porter


class MLPClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(MLPClassifierTest, self).setUp()
        self.porter = Porter(language='js')
        clf = MLPClassifier(
            activation='identity', hidden_layer_sizes=50,
            max_iter=500, alpha=1e-4, solver='sgd', tol=1e-4,
            random_state=1, learning_rate_init=.1)
        self.set_classifier(clf)

    def tearDown(self):
        super(MLPClassifierTest, self).tearDown()

    def test_hidden_activation_function_relu(self):
        clf = MLPClassifier(
            activation='relu', hidden_layer_sizes=50,
            learning_rate_init=.1)
        self.set_classifier(clf)
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            java_preds.append(self.make_pred_in_js(x))
            py_preds.append(self.make_pred_in_py(x))
        self.assertListEqual(py_preds, java_preds)

    def test_hidden_activation_function_identity(self):
        clf = MLPClassifier(
            activation='identity', hidden_layer_sizes=50,
            learning_rate_init=.1)
        self.set_classifier(clf)
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(30):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            java_preds.append(self.make_pred_in_js(x))
            py_preds.append(self.make_pred_in_py(x))
        self.assertListEqual(py_preds, java_preds)

    # def test_hidden_activation_function_tanh(self):
    #     clf = MLPClassifier(
    #         activation='tanh', hidden_layer_sizes=50,
    #         learning_rate_init=.1)
    #     self.set_classifier(clf)
    #     java_preds, py_preds = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(30):
    #         x = [random.uniform(min_vals[f], max_vals[f]) for f in
    #              range(self.n_features)]
    #         java_preds.append(self.make_pred_in_java(x))
    #         py_preds.append(self.make_pred_in_py(x))
    #
    #     matches = np.sum(np.array(java_preds) == np.array(py_preds))
    #     accuracy = matches / 30. * 100.
    #     print("accuracy: %s" % repr(accuracy))
    #     self.assertGreaterEqual(accuracy, 50.)

    # def test_hidden_activation_function_logistic(self):
    #     clf = MLPClassifier(
    #         activation='logistic', hidden_layer_sizes=50, learning_rate_init=.1)
    #     self.set_classifier(clf)
    #     java_preds, py_preds = [], []
    #     min_vals = np.amin(self.X, axis=0)
    #     max_vals = np.amax(self.X, axis=0)
    #     for n in range(150):
    #         x = [random.uniform(min_vals[f], max_vals[f]) for f in
    #              range(self.n_features)]
    #         java_preds.append(self.make_pred_in_java(x))
    #         py_preds.append(self.make_pred_in_py(x))
    #     self.assertListEqual(py_preds, java_preds)
