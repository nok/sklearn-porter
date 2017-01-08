import unittest
from ..JavaTest import JavaTest

from sklearn.naive_bayes import GaussianNB
from sklearn_porter import Porter


class GaussianNBTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(GaussianNBTest, self).setUp()
        self.porter = Porter(language='java')
        self._port_model(GaussianNB())

    def tearDown(self):
        super(GaussianNBTest, self).tearDown()
