import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.ensemble import ExtraTreesClassifier
from onl.nok.sklearn.classifier.ExtraTreesClassifier \
    import ExtraTreesClassifier as Porter


class ExtraTreesClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.clf = ExtraTreesClassifier(random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='js')
        self.create_test_files()

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
