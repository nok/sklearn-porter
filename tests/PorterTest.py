# -*- coding: utf-8 -*-

import random
import subprocess as subp
import unittest
import filecmp
import os

from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import Porter

from utils.Timer import Timer
from utils.DependencyChecker import DependencyChecker as Checker
from estimator.classifier.Classifier import Classifier
from language.Java import Java


class PorterTest(Java, Classifier, Timer, Checker, unittest.TestCase):

    DEPENDENCIES = ['mkdir', 'rm', 'java', 'javac']

    def setUp(self):
        super(PorterTest, self).setUp()
        self.load_iris_data()
        self.mdl = DecisionTreeClassifier(random_state=0)
        self._port_model()

    def tearDown(self):
        super(PorterTest, self).tearDown()

    def test_porter_args_method(self):
        """Test invalid method name."""
        args = dict(method='invalid_name')
        self.assertRaises(AttributeError, lambda: Porter(self.mdl, args))

    def test_porter_args_language(self):
        """Test invalid programming language."""
        args = dict(method='predict', language='invalid_name')
        self.assertRaises(AttributeError, lambda: Porter(self.mdl, args))

    def test_python_command_execution(self):
        """Test command line execution."""
        subp.call('rm -rf tmp'.split())
        subp.call('mkdir tmp'.split())
        filename = '{}.java'.format(self.tmp_fn)
        cp_src = os.path.join('tmp', filename)
        with open(cp_src, 'w') as f:
            porter = Porter(self.mdl)
            out = porter.export(method_name='predict', class_name=self.tmp_fn)
            f.write(out)
        # $ javac tmp/Tmp.java
        subp.call(['javac', cp_src])

        # Rename model for comparison:
        filename = '{}_2.java'.format(self.tmp_fn)
        cp_dest = os.path.join('tmp', filename)
        # $ mv tmp/Brain.java tmp/Brain_2.java
        subp.call(['mv', cp_src, cp_dest])

        # Dump model:
        filename = '{}.pkl'.format(self.tmp_fn)
        pkl_path = os.path.join('tmp', filename)
        joblib.dump(self.mdl, pkl_path)

        # Port model:
        cmd = 'python -m sklearn_porter -i {}'.format(pkl_path).split()
        subp.call(cmd)
        # Compare file contents:
        equal = filecmp.cmp(cp_src, cp_dest)

        self.assertEqual(equal, True)

    def test_java_command_execution(self):
        """Test whether the prediction of random features match or not."""
        # Create random features:
        Y, Y_py = [], []
        for n in range(self.N_RANDOM_FEATURE_SETS):
            x = [random.uniform(0., 10.) for n in range(self.n_features)]
            y_py = int(self.mdl.predict([x])[0])
            Y_py.append(y_py)
            y = self.pred_in_custom(x)
            Y.append(y)
        self.assertEqual(Y_py, Y)
