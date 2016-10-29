import unittest
from ..JavaTest import JavaTest

from sklearn.ensemble import ExtraTreesClassifier
from onl.nok.sklearn.classifier.ExtraTreesClassifier \
    import ExtraTreesClassifier as Porter


class ExtraTreesClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.clf = ExtraTreesClassifier(random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
