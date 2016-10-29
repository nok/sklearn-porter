import random
import numpy as np

import unittest
from ..JavaTest import JavaTest

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from onl.nok.sklearn.classifier.MLPClassifier \
    import MLPClassifier as Porter


class MLPClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(MLPClassifierTest, self).setUp()

        # Data:
        self.X = shuffle(self.X, random_state=0)
        self.y = shuffle(self.y, random_state=0)
        self.X_train, self.X_test,  self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.4, random_state=5)

        self.clf = MLPClassifier(
            activation='relu', hidden_layer_sizes=50, max_iter=500,
            alpha=1e-4, solver='sgd', tol=1e-4, random_state=1,
            learning_rate_init=.1)
        self.clf.fit(self.X_train, self.y_train)

        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(MLPClassifierTest, self).tearDown()
