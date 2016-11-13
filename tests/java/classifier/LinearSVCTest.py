import unittest
from ..JavaTest import JavaTest

from sklearn.svm.classes import LinearSVC
from sklearn_porter import Porter


class LinearSVCTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.porter = Porter(language='java')
        self.set_classifier(LinearSVC(C=1., random_state=0))

    def tearDown(self):
        super(LinearSVCTest, self).tearDown()
