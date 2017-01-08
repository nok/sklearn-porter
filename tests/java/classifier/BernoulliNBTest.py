import unittest
from ..JavaTest import JavaTest

from sklearn.naive_bayes import BernoulliNB
from sklearn_porter import Porter


class BernoulliNBTest(JavaTest, unittest.TestCase):

    def setUp(self):
        super(BernoulliNBTest, self).setUp()
        self.porter = Porter(language='java')
        self._port_model(BernoulliNB())

    def tearDown(self):
        super(BernoulliNBTest, self).tearDown()

    @unittest.skip('BernoulliNB is suitable for discrete data.')
    def test_random_features(self):
        pass
