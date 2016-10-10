import random
import subprocess as subp
import unittest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from onl.nok.sklearn.classifier.RandomForestClassifier import RandomForestClassifier as Porter


class RandomForestClassifierTest(unittest.TestCase):

    N_TESTS = 150

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf = RandomForestClassifier(n_estimators=100, random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)
        self.porter = Porter()

    def tearDown(self):
        self.clf = None

    def test_random_features(self):
        self._create_java_files()
        preds_from_java = []
        preds_from_py = []

        # Creating random features:
        min_vals = np.amin(self.iris.data, axis=0)
        max_vals = np.amax(self.iris.data, axis=0)
        for features in range(self.N_TESTS):
            features = [random.uniform(min_vals[f], max_vals[f]) for f in range(self.n_features)]
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self._remove_java_files()
        self.assertListEqual(preds_from_py, preds_from_java)

    def test_existing_features(self):
        self._create_java_files()
        preds_from_java = []
        preds_from_py = []

        for features in self.iris.data:
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self._remove_java_files()
        self.assertListEqual(preds_from_py, preds_from_java)

    def _create_java_files(self):
        # rm -rf temp
        subp.call(['rm', '-rf', 'temp'])
        # mkdir temp
        subp.call(['mkdir', 'temp'])

        with open('temp/' + self.tmp_fn + '.java', 'w') as file:
            main_src = self.porter.port(self.clf)
            file.write(main_src)

        # javac temp/Tmp.java
        subp.call(['javac', 'temp/' + self.tmp_fn + '.java'])

    def _remove_java_files(self):
        # rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def _make_prediction_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def _make_prediction_in_java(self, features):
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        params = [str(f).strip() for f in features]
        cmd += params
        prediction = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(prediction)

