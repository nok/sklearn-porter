import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.ensemble import RandomForestClassifier
from onl.nok.sklearn.classifier.RandomForestClassifier \
    import RandomForestClassifier as Porter


class RandomForestClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(RandomForestClassifierTest, self).setUp()
        self.clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='js')
        self.create_test_files()

    def tearDown(self):
        super(RandomForestClassifierTest, self).tearDown()
