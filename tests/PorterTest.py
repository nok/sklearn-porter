import random
import time
import subprocess as subp
import unittest
import filecmp

from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.tree import tree

from sklearn_porter import Porter


class PorterTest(unittest.TestCase):

    N_RANDOM_TESTS = 150

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.X, self.y = load_iris(return_X_y=True)
        self.n_features = len(self.X[0])
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.X, self.y)
        self._create_java_files()
        # Time:
        self.startTime = time.time()

    def tearDown(self):
        self._remove_java_files()
        self.clf = None
        # Time:
        print('%.3fs' % (time.time() - self.startTime))

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

    def _create_java_files(self):
        """Create and compile ported model for comparison of predictions."""
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'temp'])
        # $ mkdir temp
        subp.call(['mkdir', 'temp'])
        path = 'temp/%s.java' % (self.tmp_fn)
        with open(path, 'w') as file:
            porter = Porter(method_name='predict', class_name=self.tmp_fn)
            ported_model = porter.port(self.clf)
            file.write(ported_model)
        # $ javac temp/Tmp.java
        subp.call(['javac', path])

    def _remove_java_files(self):
        """Remove all temporary test files."""
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def test_data_type(self):
        """Test invalid scikit-learn model."""
        porter = Porter()
        self.assertRaises(ValueError, porter.port, '')

    def test_python_command_execution(self):
        """Test command line execution."""
        # Rename model for comparison:
        cp_src = 'temp/%s.java' % (self.tmp_fn)
        cp_dest = 'temp/%s_2.java' % (self.tmp_fn)
        # $ mv temp/Tmp.java temp/Tmp_2.java
        subp.call(['mv', cp_src, cp_dest])
        # Dump model:
        pkl_path = 'temp/%s.pkl' % (self.tmp_fn)
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
        for n in range(self.N_RANDOM_TESTS):
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
