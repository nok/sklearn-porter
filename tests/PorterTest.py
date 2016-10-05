import inspect
import random
import subprocess
import unittest
import filecmp

from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.tree import tree
from sklearn import svm

from onl.nok.sklearn.Porter import port


class PorterTest(unittest.TestCase):

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)

    def tearDown(self):
        self.clf = None

    def test_porter_param_method(self):
        self.assertRaises(AttributeError, lambda: port(self.clf, method_name="random"))

    def test_porter_param_language(self):
        self.assertRaises(AttributeError, lambda: port(self.clf, method_name="predict", language="random"))

    def _create_java_files(self):
        # rm -rf temp
        subprocess.call(['rm', '-rf', 'temp'])
        # mkdir temp
        subprocess.call(['mkdir', 'temp'])
        with open('temp/' + self.tmp_fn + '.java', 'w') as file:
            file.write(port(self.clf, method_name='predict', class_name=self.tmp_fn))
        # javac temp/Tmp.java
        subprocess.call(['javac', 'temp/' + self.tmp_fn + '.java'])

    def _remove_java_files(self):
        # rm -rf temp
        subprocess.call(['rm', '-rf', 'temp'])

    def test_data_type(self):
        self.assertRaises(ValueError, port, "")

    def test_python_command_execution(self):
        self._create_java_files()

        # mv temp/Tmp.java temp/Tmp_2.java
        subprocess.call(['mv', 'temp/' + self.tmp_fn + '.java', 'temp/' + self.tmp_fn + '_2.java'])

        joblib.dump(self.clf, 'temp/' + self.tmp_fn + '.pkl')
        # while not os.path.exists('temp/' + self.tmp_fn + '.pkl'):
        #     time.sleep(1)

        python_file = str(inspect.getfile(port)).split(".")[0] + '.py'

        # python <Porter.py> Tmp.pkl
        cmd = ['python', python_file, '-m', self.tmp_fn + '.pkl']
        subprocess.call(cmd, cwd='temp')

        equal = filecmp.cmp('temp/' + self.tmp_fn + '.java', 'temp/' + self.tmp_fn + '_2.java')

        self._remove_java_files()
        self.assertEqual(equal, True)

    def test_java_command_execution(self):
        self._create_java_files()

        python_predictions = []
        java_predictions = []

        # Create random features:
        for features in range(150):
            features = [random.uniform(0., 10.) for f in range(self.n_features)]
            python_prediction = int(self.clf.predict([features])[0])
            python_predictions.append(python_prediction)
            java_prediction = self._make_prediction_in_java(features)
            java_predictions.append(java_prediction)

        self._remove_java_files()
        self.assertEqual(python_predictions, java_predictions)

    def _make_prediction_in_java(self, features):
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        params = [str(f).strip() for f in features]
        cmd += params
        prediction = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return int(prediction)
