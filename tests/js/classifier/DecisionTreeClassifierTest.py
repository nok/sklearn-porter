import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.tree import DecisionTreeClassifier
from onl.nok.sklearn.Porter import Porter


class DecisionTreeClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.porter = Porter(language='js')
        clf = DecisionTreeClassifier(random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
