import unittest
from ..JavaTest import JavaTest

from sklearn.tree import DecisionTreeClassifier
from onl.nok.sklearn.Porter import Porter


class DecisionTreeClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.porter = Porter(language='java')
        self.set_classifier(DecisionTreeClassifier(random_state=0))

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
