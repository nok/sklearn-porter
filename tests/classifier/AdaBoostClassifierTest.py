import random
import subprocess
import unittest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.tree import DecisionTreeClassifier as DT

from onl.nok.sklearn.classifier.AdaBoostClassifier import AdaBoostClassifier


class AdaBoostClassifierTest(unittest.TestCase):

    TESTS = 150

    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        base_estimator = DT(max_depth=4, random_state=0)
        self.clf = ADA(base_estimator=base_estimator, n_estimators=100, random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)

    def tearDown(self):
        self.clf = None

    # @unittest.expectedFailure
    def test_random_features(self):
        self._create_java_files()
        preds_from_java = []
        preds_from_py = []

        # Creating random features:
        min_vals = np.amin(self.iris.data, axis=0)
        max_vals = np.amax(self.iris.data, axis=0)
        for features in range(self.TESTS):
            features = [random.uniform(min_vals[f], max_vals[f]) for f in range(self.n_features)]
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        # Debugging: Compare results manually
        # for i, el in enumerate(preds_from_py):
        #     print("%d - %d - %d - %s" % (
        #         i,
        #         preds_from_py[i],
        #         preds_from_java[i],
        #         str(preds_from_py[i] == preds_from_java[i])))

        # errors = 0
        # for i, el in enumerate(preds_from_py):
        #     if (preds_from_py[i] != preds_from_java[i]):
        #         errors += 1
        # print('Rounding precision error: %f %%' % (float(errors) / float(self.TESTS) * 100.0))

        self._remove_java_files()
        # self.assertTrue(errors < 15)
        self.assertListEqual(preds_from_py, preds_from_java)

    def test_existing_features(self):
        self._create_java_files()
        preds_from_java = []
        preds_from_py = []

        for features in self.iris.data:
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        # Debugging: Compare results manually
        # for i, el in enumerate(preds_from_py):
        #     print("%d - %d - %d - %s" % (
        #         i,
        #         preds_from_py[i],
        #         preds_from_java[i],
        #         str(preds_from_py[i] == preds_from_java[i])))

        self._remove_java_files()
        self.assertListEqual(preds_from_py, preds_from_java)

    def _create_java_files(self):
        # rm -rf temp
        subprocess.call(['rm', '-rf', 'temp'])
        # mkdir temp
        subprocess.call(['mkdir', 'temp'])

        with open('temp/' + self.tmp_fn + '.java', 'w') as file:
            porter = AdaBoostClassifier()
            main_src = porter.port(self.clf)
            file.write(main_src)

        # javac temp/Tmp.java
        subprocess.call(['javac', 'temp/' + self.tmp_fn + '.java'])

    def _remove_java_files(self):
        # rm -rf temp
        subprocess.call(['rm', '-rf', 'temp'])

    def _make_prediction_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def _make_prediction_in_java(self, features):
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        params = [str(f).strip() for f in features]
        cmd += params
        prediction = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return int(prediction)

