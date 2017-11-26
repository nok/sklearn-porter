# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from sklearn.svm.classes import SVC
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

from tests.estimator.classifier.Classifier import Classifier
from tests.language.JavaScript import JavaScript


class SVCJSTest(JavaScript, Classifier, TestCase):

    def setUp(self):
        super(SVCJSTest, self).setUp()
        self.estimator = SVC(C=1., kernel='rbf',
                             gamma=0.001, random_state=0)

    def tearDown(self):
        super(SVCJSTest, self).tearDown()

    def test_linear_kernel(self):
        self.estimator = SVC(C=1., kernel='linear',
                             gamma=0.001, random_state=0)
        self.load_iris_data()
        self._port_estimator()
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

    def test_sigmoid_kernel(self):
        self.estimator = SVC(C=1., kernel='sigmoid',
                             gamma=0.001, random_state=0)
        self.load_iris_data()
        self._port_estimator()
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

    def test_pipeline_estimator(self):
        self.X, self.y = samples_generator.make_classification(
            n_informative=5, n_redundant=0, random_state=42)
        anova_filter = SelectKBest(f_regression, k=5)
        self.estimator = Pipeline([('anova', anova_filter), ('svc', SVC(kernel='linear'))])
        self.estimator.set_params(anova__k=10, svc__C=.1)
        try:
            self._port_estimator()
        except Exception as e:
            self.fail('Unexpected exception raised: {}'.format(e.message))
        finally:
            self._clear_estimator()
