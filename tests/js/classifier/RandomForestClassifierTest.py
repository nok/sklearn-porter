import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.ensemble import RandomForestClassifier
from sklearn_porter import Porter


class RandomForestClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(RandomForestClassifierTest, self).setUp()
        self.porter = Porter(language='js')
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(RandomForestClassifierTest, self).tearDown()
