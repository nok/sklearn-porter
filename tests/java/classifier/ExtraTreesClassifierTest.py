import unittest
from ..JavaTest import JavaTest

from sklearn.ensemble import ExtraTreesClassifier
from onl.nok.sklearn.classifier.ExtraTreesClassifier \
    import ExtraTreesClassifier as Porter


class ExtraTreesClassifierTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.porter = Porter(language='java')
        self.set_classifier(ExtraTreesClassifier(random_state=0))

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
