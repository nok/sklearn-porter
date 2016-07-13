import random
import subprocess
from unittest import TestCase
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as DT

from onl.nok.sklearn.export.classifier.DecisionTreeClassifier import DecisionTreeClassifier


class DecisionTreeClassifierTest(TestCase):


    def setUp(self):
        self.tmp_fn = 'Tmp'
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf = DT(random_state=0)
        self.clf.fit(self.iris.data, self.iris.target)


    def tearDown(self):
        del self.clf


    def test_random_features(self):
        self._create_java_files()

        preds_from_java = []
        preds_from_py = []

        # Creating random features:
        for features in range(150):
            features = [random.uniform(0., 10.) for f in range(self.n_features)]
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self._remove_java_files()
        self.assertEqual(preds_from_py, preds_from_java)


    def test_existing_features(self):
        self._create_java_files()

        preds_from_java = []
        preds_from_py = []

        # Getting existing features:
        for features in self.iris.data:
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self._remove_java_files()
        self.assertEqual(preds_from_py, preds_from_java)


    def _create_java_files(self):
        # Porting to Java:
        with open(self.tmp_fn + '.java', 'w') as file:
            main_src = DecisionTreeClassifier.predict(self.clf)
            file.write(main_src)
        # Compiling Java test class:
        subprocess.call(['javac', self.tmp_fn + '.java'])


    def _remove_java_files(self):
        subprocess.call(['rm', self.tmp_fn + '.class'])
        subprocess.call(['rm', self.tmp_fn + '.java'])


    def _make_prediction_in_py(self, features):
        return int(self.clf.predict([features])[0])


    def _make_prediction_in_java(self, features):
        execution = ['java', self.tmp_fn]
        params = [str(f).strip() for f in features]
        command = execution + params
        return int(subprocess.check_output(command).strip())

