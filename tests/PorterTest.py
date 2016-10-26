import inspect
import random
import subprocess as subp
import unittest
import filecmp

from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.tree import tree
from sklearn import svm

from onl.nok.sklearn.Porter import port


class PorterTest(unittest.TestCase):

    N_TESTS = 150

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)
        self._create_java_files()

    def tearDown(self):
        self._remove_java_files()
        self.clf = None

    def test_porter_args_method(self):
        args = dict(method_name="random")
        self.assertRaises(AttributeError, lambda: port(self.clf, args))

    def test_porter_args_language(self):
        args = dict(method_name="predict", language="random")
        self.assertRaises(AttributeError, lambda: port(self.clf, args))

    def _create_java_files(self):
        # rm -rf temp
        subp.call(['rm', '-rf', 'temp'])
        # mkdir temp
        subp.call(['mkdir', 'temp'])
        path = 'temp/%s.java' % (self.tmp_fn)
        with open(path, 'w') as file:
            out = port(self.clf, method_name='predict', class_name=self.tmp_fn)
            file.write(out)
        # javac temp/Tmp.java
        subp.call(['javac', path])

    def _remove_java_files(self):
        # rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def test_data_type(self):
        self.assertRaises(ValueError, port, "")

    def test_python_command_execution(self):
        # Rename model for comparison:
        cp_src = 'temp/%s.java' % (self.tmp_fn)
        cp_dest = 'temp/%s_2.java' % (self.tmp_fn)
        # mv temp/Tmp.java temp/Tmp_2.java
        subp.call(['mv', cp_src, cp_dest])

        # Dump model:
        joblib.dump(self.clf, 'temp/%s.pkl' % (self.tmp_fn))

        # Port model:
        porter_path = str(inspect.getfile(port)).split(".")[0] + '.py'
        # python <Porter.py> Tmp.pkl
        cmd = ['python', porter_path, '-m', self.tmp_fn + '.pkl']
        subp.call(cmd, cwd='temp')

        # Compare file content:
        equal = filecmp.cmp(cp_src, cp_dest)
        self.assertEqual(equal, True)

    def test_java_command_execution(self):
        # Create random features:
        java_preds, py_preds = [], []
        for n in range(self.N_TESTS):
            x = [random.uniform(0., 10.) for n in range(self.n_features)]
            py_pred = int(self.clf.predict([x])[0])
            py_preds.append(py_pred)
            java_pred = self.make_pred_in_java(x)
            java_preds.append(java_pred)

        self.assertEqual(py_preds, java_preds)

    def make_pred_in_java(self, features):
        # -> java -classpath temp <temp_filename> <features>
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
