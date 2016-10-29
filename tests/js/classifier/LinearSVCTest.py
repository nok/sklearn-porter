import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.svm.classes import LinearSVC
from onl.nok.sklearn.classifier.LinearSVC \
    import LinearSVC as Porter


class LinearSVCTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.clf = LinearSVC(C=1., random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='js')
        self.create_test_files()

    def tearDown(self):
        super(LinearSVCTest, self).tearDown()
