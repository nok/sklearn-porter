import unittest
from ..JavaTest import JavaTest

from sklearn.svm.classes import SVC
from onl.nok.sklearn.classifier.SVC \
    import SVC as Porter


class SVCTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(SVCTest, self).setUp()
        self.clf = SVC(C=1., kernel='rbf', gamma=0.001, random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(SVCTest, self).tearDown()
