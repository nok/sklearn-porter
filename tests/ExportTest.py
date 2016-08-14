import inspect
import random
import subprocess
import unittest

from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.tree import tree

from onl.nok.sklearn.export import export


class ExportTest(unittest.TestCase):

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)

    def tearDown(self):
        self.clf = None


    def _create_java_files(self):
        # Porting to Java:
        with open(self.tmp_fn + '.java', 'w') as file:
            file.write(export(self.clf, method_name='predict', class_name=self.tmp_fn))
        # Compiling Java test class:
        subprocess.call(['javac', self.tmp_fn + '.java'])

    def _remove_java_files(self):
        subprocess.call(['rm', self.tmp_fn + '.class'])
        subprocess.call(['rm', self.tmp_fn + '.java'])


    def test_data_type(self):
        self.assertRaises(ValueError, export, "")


    def test_command_execution(self):
        self._create_java_files()

        joblib.dump(self.clf, self.tmp_fn + '.pkl')
        python_file = str(inspect.getfile(export)).split(".")[0] + '.py'

        subprocess.call(['python', python_file, 'Tmp.pkl', '--output', 'Tmp.java'])
        subprocess.call(['javac', self.tmp_fn + '.java'])

        preds_from_java = []
        preds_from_py = []

        # Creating random features:
        for features in range(150):
            features = [random.uniform(0., 10.) for f in range(self.n_features)]
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        subprocess.call(['rm', self.tmp_fn + '.pkl'])
        subprocess.call(['rm', self.tmp_fn + '.pkl_01.npy'])
        subprocess.call(['rm', self.tmp_fn + '.pkl_02.npy'])
        subprocess.call(['rm', self.tmp_fn + '.pkl_03.npy'])
        subprocess.call(['rm', self.tmp_fn + '.pkl_04.npy'])

        self._remove_java_files()
        self.assertEqual(preds_from_py, preds_from_java)


    def _make_prediction_in_py(self, features):
        return int(self.clf.predict([features])[0])


    def _make_prediction_in_java(self, features):
        execution = ['java', self.tmp_fn]
        params = [str(f).strip() for f in features]
        command = execution + params
        return int(subprocess.check_output(command).strip())