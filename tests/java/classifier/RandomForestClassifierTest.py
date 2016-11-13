import unittest
from ..JavaTest import JavaTest

from sklearn.ensemble import RandomForestClassifier
from sklearn_porter import Porter


class RandomForestClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(RandomForestClassifierTest, self).setUp()
        self.porter = Porter(language='java')
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(RandomForestClassifierTest, self).tearDown()
