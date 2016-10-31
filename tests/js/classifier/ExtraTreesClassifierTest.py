import unittest
from ..JavaScriptTest import JavaScriptTest

from sklearn.ensemble import ExtraTreesClassifier
from onl.nok.sklearn.classifier.ExtraTreesClassifier \
    import ExtraTreesClassifier as Porter


class ExtraTreesClassifierTest(JavaScriptTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.porter = Porter(language='js')
        clf = ExtraTreesClassifier(random_state=0)
        self.set_classifier(clf)

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
