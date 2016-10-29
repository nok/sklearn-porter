import unittest
from ..JavaTest import JavaTest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from onl.nok.sklearn.classifier.AdaBoostClassifier \
    import AdaBoostClassifier as Porter


class AdaBoostClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(AdaBoostClassifierTest, self).setUp()
        base_estimator = DecisionTreeClassifier(max_depth=4, random_state=0)
        self.clf = AdaBoostClassifier(
            base_estimator=base_estimator, n_estimators=100, random_state=0)
        self.clf.fit(self.X, self.y)
        self.porter = Porter(language='java')
        self.create_test_files()

    def tearDown(self):
        super(AdaBoostClassifierTest, self).tearDown()
