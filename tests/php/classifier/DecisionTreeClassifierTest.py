import unittest
from ..PhpTest import PhpTest

from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter


class DecisionTreeClassifierTest(PhpTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.porter = Porter(language='php')
        clf = DecisionTreeClassifier(random_state=0)
        self._port_model(clf)

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
