# -*- coding: utf-8 -*-

import random
import unittest
import filecmp
import os

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

from sklearn_porter.Porter import Porter
from sklearn_porter.utils.Shell import Shell

from tests.utils.Timer import Timer
from tests.utils.DependencyChecker import DependencyChecker as Checker
from tests.estimator.classifier.Classifier import Classifier
from tests.language.Java import Java


class PorterTest(Java, Classifier, Timer, Checker, unittest.TestCase):

    DEPENDENCIES = ['mkdir', 'rm', 'java', 'javac']

    def setUp(self):
        super(PorterTest, self).setUp()
        self.load_iris_data()
        self.estimator = LinearSVC(C=1., random_state=0)
        self._port_estimator()

    def tearDown(self):
        super(PorterTest, self).tearDown()

    def test_porter_args_method(self):
        """Test invalid method name."""
        self.assertRaises(AttributeError, lambda: Porter(self.estimator,
                                                         method='invalid'))

    def test_porter_args_language(self):
        """Test invalid programming language."""
        self.assertRaises(AttributeError, lambda: Porter(self.estimator,
                                                         language='invalid'))

    def test_python_command_execution(self):
        """Test command line execution."""
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        filename = '{}.java'.format(self.tmp_fn)
        cp_src = os.path.join('tmp', filename)
        with open(cp_src, 'w') as f:
            porter = Porter(self.estimator)
            out = porter.export(method_name='predict', class_name=self.tmp_fn)
            f.write(out)
        # $ javac tmp/Tmp.java
        Shell.call(['javac', cp_src])

        # Rename estimator for comparison:
        filename = '{}_2.java'.format(self.tmp_fn)
        cp_dest = os.path.join('tmp', filename)
        # $ mv tmp/Brain.java tmp/Brain_2.java
        Shell.call(['mv', cp_src, cp_dest])

        # Dump estimator:
        filename = '{}.pkl'.format(self.tmp_fn)
        pkl_path = os.path.join('tmp', filename)
        joblib.dump(self.estimator, pkl_path)

        # Port estimator:
        cmd = 'python -m sklearn_porter -i {} --class_name Brain'.format(pkl_path)
        Shell.call(cmd)
        # Compare file contents:
        equal = filecmp.cmp(cp_src, cp_dest)

        self.assertEqual(equal, True)

    def test_java_command_execution(self):
        """Test whether the prediction of random features match or not."""
        # Create random features:
        Y, Y_py = [], []
        for n in range(self.N_RANDOM_FEATURE_SETS):
            x = [random.uniform(0., 10.) for n in range(self.n_features)]
            y_py = int(self.estimator.predict([x])[0])
            Y_py.append(y_py)
            y = self.pred_in_custom(x)
            Y.append(y)
        self.assertEqual(Y_py, Y)

    def test_filename_generation_for_java(self):
        language = 'java'
        self.assertEqual(Porter._get_filename('mdl', language), 'Mdl.java')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'Mdl.java')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.java')

    def test_filename_generation_for_php(self):
        language = 'php'
        self.assertEqual(Porter._get_filename('mdl', language), 'Mdl.php')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'Mdl.php')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.php')

    def test_filename_generation_for_c(self):
        language = 'c'
        self.assertEqual(Porter._get_filename('mdl', language), 'mdl.c')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'mdl.c')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.c')

    def test_filename_generation_for_js(self):
        language = 'js'
        self.assertEqual(Porter._get_filename('mdl', language), 'mdl.js')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'mdl.js')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.js')

    def test_filename_generation_for_go(self):
        language = 'go'
        self.assertEqual(Porter._get_filename('mdl', language), 'mdl.go')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'mdl.go')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.go')

    def test_filename_generation_for_ruby(self):
        language = 'ruby'
        self.assertEqual(Porter._get_filename('mdl', language), 'mdl.rb')
        self.assertEqual(Porter._get_filename(' mdl ', language), 'mdl.rb')
        self.assertEqual(Porter._get_filename('MDL', language), 'MDL.rb')

    def test_commands_generation_for_java(self):
        comp_cmd, exec_cmd = Porter._get_commands('Mdl.java', 'Mdl', 'java')
        self.assertEqual(comp_cmd, 'javac Mdl.java')
        self.assertEqual(exec_cmd, 'java -classpath . Mdl')

    def test_commands_generation_for_php(self):
        comp_cmd, exec_cmd = Porter._get_commands('Mdl.php', 'Mdl', 'php')
        self.assertEqual(comp_cmd, None)
        self.assertEqual(exec_cmd, 'php -f Mdl.php')

    def test_commands_generation_for_c(self):
        comp_cmd, exec_cmd = Porter._get_commands('mdl.c', 'mdl', 'c')
        self.assertEqual(comp_cmd, 'gcc mdl.c -lm -o mdl')
        self.assertEqual(exec_cmd, './mdl')

    def test_commands_generation_for_js(self):
        comp_cmd, exec_cmd = Porter._get_commands('mdl.js', 'mdl', 'js')
        self.assertEqual(comp_cmd, None)
        self.assertEqual(exec_cmd, 'node mdl.js')

    def test_commands_generation_for_go(self):
        comp_cmd, exec_cmd = Porter._get_commands('mdl.go', 'mdl', 'go')
        self.assertEqual(comp_cmd, 'go build -o mdl mdl.go')
        self.assertEqual(exec_cmd, './mdl')