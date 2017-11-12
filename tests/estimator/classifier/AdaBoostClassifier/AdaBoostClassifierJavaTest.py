# -*- coding: utf-8 -*-

import unittest
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class AdaBoostClassifierJavaTest(Java, Classifier, unittest.TestCase):

    def setUp(self):
        super(AdaBoostClassifierJavaTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4,
                                                random_state=0)
        self.estimator = AdaBoostClassifier(base_estimator=base_estimator,
                                            n_estimators=100, random_state=0)

    def tearDown(self):
        super(AdaBoostClassifierJavaTest, self).tearDown()

    def test_random_features_w_binary_data_export(self):
        self.load_binary_data()
        self._port_estimator(export_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_random_features_w_iris_data_export(self):
        self.load_iris_data()
        self._port_estimator(export_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_random_features_w_digits_data_export(self):
        self.load_digits_data()
        self._port_estimator(export_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features_w_binary_data_export(self):
        self.load_binary_data()
        self._port_estimator(export_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features_w_iris_data_export(self):
        self.load_iris_data()
        self._port_estimator(export_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features_w_digits_data_export(self):
        self.load_digits_data()
        self._port_estimator(export_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x, export_data=True))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)