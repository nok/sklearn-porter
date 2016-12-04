import unittest
from ..CTest import CTest

from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter


class ExtraTreesClassifierTest(CTest, unittest.TestCase):

    def setUp(self):
        super(ExtraTreesClassifierTest, self).setUp()
        self.porter = Porter(language='c')
        self._port_model(ExtraTreesClassifier(random_state=0))

    def tearDown(self):
        super(ExtraTreesClassifierTest, self).tearDown()
