# -*- coding: utf-8 -*-

from unittest import TestCase
import numpy as np
import random

from sklearn.svm.classes import NuSVC

from ..Classifier import Classifier
from ...language.JavaScript import JavaScript as JS


class NuSVCJSTest(JS, Classifier, TestCase):

    def setUp(self):
        super(NuSVCJSTest, self).setUp()
        self.mdl = NuSVC(kernel='rbf', gamma=0.001, random_state=0)

    def tearDown(self):
        super(NuSVCJSTest, self).tearDown()
