import unittest
from ..JavaTest import JavaTest

from sklearn.tree import DecisionTreeClassifier
from onl.nok.sklearn.classifier.DecisionTreeClassifier \
    import DecisionTreeClassifier as Porter


class DecisionTreeClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(DecisionTreeClassifierTest, self).setUp()
        self.clf = DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(DecisionTreeClassifierTest, self).tearDown()
