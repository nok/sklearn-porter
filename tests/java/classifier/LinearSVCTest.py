import unittest
from ..JavaTest import JavaTest

from sklearn.svm.classes import LinearSVC
from onl.nok.sklearn.classifier.LinearSVC \
    import LinearSVC as Porter


class LinearSVCTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.clf = LinearSVC(C=1., random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(LinearSVCTest, self).tearDown()
