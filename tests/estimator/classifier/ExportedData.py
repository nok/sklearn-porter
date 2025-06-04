# -*- coding: utf-8 -*-

import numpy as np


class ExportedData():

    pass

    # def test_random_features__binary_data__exported(self):
    #     self.load_binary_data()
    #     self._port_estimator(export_data=True)
    #     amin = np.amin(self.X, axis=0)
    #     amax = np.amax(self.X, axis=0)
    #     shape = (self.TEST_N_RANDOM_FEATURE_SETS, self.n_features)
    #     X = np.random.uniform(low=amin, high=amax, size=shape)
    #     Y_py = self.estimator.predict(X).tolist()
    #     Y = [self.pred_in_custom(x, export_data=True) for x in X]
    #     self._clear_estimator()
    #     self.assertListEqual(Y, Y_py)

    # def test_random_features__iris_data__exported(self):
    #     self.load_iris_data()
    #     self._port_estimator(export_data=True)
    #     amin = np.amin(self.X, axis=0)
    #     amax = np.amax(self.X, axis=0)
    #     shape = (self.TEST_N_RANDOM_FEATURE_SETS, self.n_features)
    #     X = np.random.uniform(low=amin, high=amax, size=shape)
    #     Y_py = self.estimator.predict(X).tolist()
    #     Y = [self.pred_in_custom(x, export_data=True) for x in X]
    #     self._clear_estimator()
    #     self.assertListEqual(Y, Y_py)

    # def test_random_features__digits_data__exported(self):
    #     self.load_digits_data()
    #     self._port_estimator(export_data=True)
    #     amin = np.amin(self.X, axis=0)
    #     amax = np.amax(self.X, axis=0)
    #     shape = (self.TEST_N_RANDOM_FEATURE_SETS, self.n_features)
    #     X = np.random.uniform(low=amin, high=amax, size=shape)
    #     Y_py = self.estimator.predict(X).tolist()
    #     Y = [self.pred_in_custom(x, export_data=True) for x in X]
    #     self._clear_estimator()
    #     self.assertListEqual(Y, Y_py)

    # def test_existing_features__binary_data__exported(self):
    #     self.load_binary_data()
    #     self._port_estimator(export_data=True)
    #     preds, ground_truth = [], []
    #     n = min(self.TEST_N_EXISTING_FEATURE_SETS, len(self.X))
    #     for x in self.X[:n]:
    #         preds.append(self.pred_in_custom(x, export_data=True))
    #         ground_truth.append(self.pred_in_py(x))
    #     self._clear_estimator()
    #     # noinspection PyUnresolvedReferences
    #     self.assertListEqual(preds, ground_truth)

    # def test_existing_features__iris_data__exported(self):
    #     self.load_iris_data()
    #     self._port_estimator(export_data=True)
    #     preds, ground_truth = [], []
    #     n = min(self.TEST_N_EXISTING_FEATURE_SETS, len(self.X))
    #     for x in self.X[:n]:
    #         preds.append(self.pred_in_custom(x, export_data=True))
    #         ground_truth.append(self.pred_in_py(x))
    #     self._clear_estimator()
    #     # noinspection PyUnresolvedReferences
    #     self.assertListEqual(preds, ground_truth)

    # def test_existing_features__digits_data__exported(self):
    #     self.load_digits_data()
    #     self._port_estimator(export_data=True)
    #     preds, ground_truth = [], []
    #     n = min(self.TEST_N_EXISTING_FEATURE_SETS, len(self.X))
    #     for x in self.X[:n]:
    #         preds.append(self.pred_in_custom(x, export_data=True))
    #         ground_truth.append(self.pred_in_py(x))
    #     self._clear_estimator()
    #     # noinspection PyUnresolvedReferences
    #     self.assertListEqual(preds, ground_truth)
