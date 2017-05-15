# -*- coding: utf-8 -*-

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle


class Classifier(object):

    def setUp(self):
        samples = load_diabetes()
        self.X = shuffle(samples.data, random_state=0)
        self.y = shuffle(samples.target, random_state=0)
        self.n_features = len(self.X[0])
