# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import LinearSVC

from sklearn_porter import Porter
from ..Classifier import Classifier
from ...language.Java import Java


class LinearSVCJavaTest(Java, Classifier, TestCase):

    def setUp(self):
        super(LinearSVCJavaTest, self).setUp()
        self.mdl = LinearSVC(C=1., random_state=0)

    def tearDown(self):
        super(LinearSVCJavaTest, self).tearDown()

    def test_model_within_optimizer(self):
        pipe = Pipeline([
            ('reduce_dim', PCA()),
            ('classify', LinearSVC())
        ])
        n_features_options = [2, 4, 8]
        c_options = [1, 10, 100, 1000]
        param_grid = [
            {
                'reduce_dim': [PCA(iterated_power=7), NMF()],
                'reduce_dim__n_components': n_features_options,
                'classify__C': c_options
            },
            {
                'reduce_dim': [SelectKBest(chi2)],
                'reduce_dim__k': n_features_options,
                'classify__C': c_options
            },
        ]
        grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
        digits = load_digits()
        grid.fit(digits.data, digits.target)

        try:
            Porter(grid, language='java')
        except ValueError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)
