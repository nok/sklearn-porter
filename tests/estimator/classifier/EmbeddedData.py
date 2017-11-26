# -*- coding: utf-8 -*-

import numpy as np


class EmbeddedData():

    def test_random_features__binary_data__embedded(self):
        self.load_binary_data()
        self._port_estimator(embed_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_random_features__iris_data__embedded(self):
        self.load_iris_data()
        self._port_estimator(embed_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_random_features__digits_data__embedded(self):
        self.load_digits_data()
        self._port_estimator(embed_data=True)
        amin = np.amin(self.X, axis=0)
        amax = np.amax(self.X, axis=0)
        preds, ground_truth = [], []
        for _ in range(self.N_RANDOM_FEATURE_SETS):
            x = np.random.uniform(amin, amax, self.n_features)
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features__binary_data__embedded(self):
        self.load_binary_data()
        self._port_estimator(embed_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features__iris_data__embedded(self):
        self.load_iris_data()
        self._port_estimator(embed_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)

    def test_existing_features__digits_data__embedded(self):
        self.load_digits_data()
        self._port_estimator(embed_data=True)
        preds, ground_truth = [], []
        n = min(self.N_EXISTING_FEATURE_SETS, len(self.X))
        for x in self.X[:n]:
            preds.append(self.pred_in_custom(x))
            ground_truth.append(self.pred_in_py(x))
        self._clear_estimator()
        # noinspection PyUnresolvedReferences
        self.assertListEqual(preds, ground_truth)