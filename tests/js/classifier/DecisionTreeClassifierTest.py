import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.tree import DecisionTreeClassifier
from onl.nok.sklearn.classifier.DecisionTreeClassifier \
    import DecisionTreeClassifier as Porter


class DecisionTreeClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.clf = DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='js')
        self.create_test_files()

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
