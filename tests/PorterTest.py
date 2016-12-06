import random
import time
import subprocess as subp
import unittest
import filecmp
import os

from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.tree import tree

from sklearn_porter import Porter


class PorterTest(unittest.TestCase):

    TEST_DEPENDENCIES = ['mkdir', 'rm', 'java', 'javac']

    def setUp(self):
        self._check_test_dependencies()
        self._init_test()
        self._train_model()
        self._port_model()
        self._start_test()

    def tearDown(self):
        self._clear_model()
        self._stop_test()

    def _check_test_dependencies(self):
        # $ if hash gcc 2/dev/null; then echo 1; else echo 0; fi
        for dep in self.TEST_DEPENDENCIES:
            cmd = 'if hash %s 2/dev/null; then echo 1; else echo 0; fi' % dep
            available = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                err_msg = ('The required test dependency \'{0}\' '
                           'is not available.').format(dep)
                self.fail(err_msg)

    def _init_test(self):
        self.tmp_fn = 'Tmp'
        self.n_random_tests = 150
        if 'N_RANDOM_TESTS' in set(os.environ):
            n = os.environ.get('N_RANDOM_TESTS')
            if str(n).strip().isdigit():
                if int(n) > 0:
                    self.n_random_tests = int(n)

    def _train_model(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.n_features = len(self.X[0])
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.X, self.y)

    def _start_test(self):
        self.startTime = time.time()

    def _stop_test(self):
        print('%.3fs' % (time.time() - self.startTime))

    def _port_model(self):
        """Create and compile ported model for comparison of predictions."""
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'temp'])
        # $ mkdir temp
        subp.call(['mkdir', 'temp'])
        filename = '%s.java' % self.tmp_fn
        path = os.path.join('temp', filename)
        with open(path, 'w') as f:
            porter = Porter(method_name='predict', class_name=self.tmp_fn)
            ported_model = porter.port(self.clf)
            f.write(ported_model)
        # $ javac temp/Tmp.java
        subp.call(['javac', path])

    def _clear_model(self):
        """Remove all temporary test files."""
        self.clf = None
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def test_porter_args_method(self):
        """Test invalid method name."""
        args = dict(method_name='random')
        porter = Porter(args)
        self.assertRaises(AttributeError, lambda: porter.port(self.clf))

    def test_porter_args_language(self):
        """Test invalid programming language."""
        args = dict(method_name='predict', language='random')
        porter = Porter(args)
        self.assertRaises(AttributeError, lambda: porter.port(self.clf))

    def test_data_type(self):
        """Test invalid scikit-learn model."""
        porter = Porter()
        self.assertRaises(ValueError, porter.port, '')

    def test_python_command_execution(self):
        """Test command line execution."""
        # Rename model for comparison:
        filename = '%s.java' % self.tmp_fn
        cp_src = os.path.join('temp', filename)
        filename = '%s_2.java' % self.tmp_fn
        cp_dest = os.path.join('temp', filename)
        # $ mv temp/Tmp.java temp/Tmp_2.java
        subp.call(['mv', cp_src, cp_dest])
        # Dump model:
        filename = '%s.pkl' % self.tmp_fn
        pkl_path = os.path.join('temp', filename)
        joblib.dump(self.clf, pkl_path)
        # Port model:
        cmd = ['python', '-m', 'sklearn_porter', '-i', pkl_path]
        subp.call(cmd)
        # Compare file contents:
        equal = filecmp.cmp(cp_src, cp_dest)
        self.assertEqual(equal, True)

    def test_java_command_execution(self):
        """Test whether the prediction of random features match or not."""
        # Create random features:
        java_preds, py_preds = [], []
        for n in range(self.n_random_tests):
            x = [random.uniform(0., 10.) for n in range(self.n_features)]
            py_pred = int(self.clf.predict([x])[0])
            py_preds.append(py_pred)
            java_pred = self.make_pred_in_java(x)
            java_preds.append(java_pred)
        self.assertEqual(py_preds, java_preds)

    def make_pred_in_java(self, features):
        """Run Java prediction on the command line."""
        # $ java -classpath temp <temp_filename> <features>
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
