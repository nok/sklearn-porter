import random
import time
import subprocess as subp
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils import shuffle


class JavaTest():

    LANGUAGE = 'java'
    N_RANDOM_TESTS = 150

    # noinspection PyPep8Naming
    def setUp(self):
        self.tmp_fn = 'Tmp'
        # Data:
        self.X, self.y = load_iris(return_X_y=True)
        self.X = shuffle(self.X, random_state=0)
        self.y = shuffle(self.y, random_state=0)
        self.n_features = len(self.X[0])
        # Placeholders:
        self.clf = None
        self.porter = None
        # self.create_test_files()
        self.startTime = time.time()

    # noinspection PyPep8Naming
    def tearDown(self):
        self.remove_test_files()
        self.clf = None
        print('%.3fs' % (time.time() - self.startTime))

    def set_classifier(self, clf):
        self.clf = clf
        self.clf.fit(self.X, self.y)
        self.create_test_files()

    def test_model_support(self):
        is_classifier = self.porter.is_supported_classifier(self.clf)
        is_regressor = self.porter.is_supported_regressor(self.clf)
        is_model_valid = is_classifier or is_regressor
        # noinspection PyUnresolvedReferences
        self.assertTrue(is_model_valid)

    def test_random_features(self):
        # Creating random features:
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.N_RANDOM_TESTS):
            x = [random.uniform(min_vals[f], max_vals[f])
                 for f in range(self.n_features)]
            java_preds.append(self.make_pred_in_java(x))
            py_preds.append(self.make_pred_in_py(x))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(py_preds, java_preds)

    def test_existing_features(self):
        # Get existing features:
        java_preds, py_preds = [], []
        for X in self.X:
            java_preds.append(self.make_pred_in_java(X))
            py_preds.append(self.make_pred_in_py(X))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(java_preds, py_preds)

    def create_test_files(self):
        self.remove_test_files()
        # Create the temporary test dictionary:
        # $ mkdir temp
        subp.call(['mkdir', 'temp'])
        # Save transpiled model:
        path = 'temp/' + self.tmp_fn + '.java'
        with open(path, 'w') as file:
            file.write(self.porter.port(self.clf))
        # Compile model:
        # $ javac temp/Tmp.java
        subp.call(['javac', path])

    def remove_test_files(self):
        # Remove the temporary test directory:
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def make_pred_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def make_pred_in_java(self, features):
        # $ java -classpath temp <temp_filename> <features>
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
