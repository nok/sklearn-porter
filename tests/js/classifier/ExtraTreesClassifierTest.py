import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter


class ExtraTreesClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.porter = Porter(language='js')
        clf = ExtraTreesClassifier(random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
