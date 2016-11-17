import unittest
from ..CTest import CTest

from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter


class DecisionTreeClassifierTest(CTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.porter = Porter(language='c')
        self.set_classifier(DecisionTreeClassifier(random_state=0))

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
