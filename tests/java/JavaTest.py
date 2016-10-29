import random
import subprocess as subp
import numpy as np

from sklearn.datasets import load_iris


class JavaTest():

    N_RANDOM_TESTS = 150

    # noinspection PyPep8Naming
    def setUp(self):
        self.tmp_fn = 'Tmp'

        # Data:
        self.X, self.y = load_iris(return_X_y=True)
        self.n_features = len(self.X[0])

        # Placeholders:
        self.clf = None
        self.porter = None
        # self.create_test_files()

    # noinspection PyPep8Naming
    def tearDown(self):
        self.remove_test_files()
        self.clf = None

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
        # -> mkdir temp
        subp.call(['mkdir', 'temp'])

        # Save transpiled model:
        path = 'temp/' + self.tmp_fn + '.java'
        with open(path, 'w') as file:
            file.write(self.porter.port(self.clf))

        # Compile model:
        # -> javac temp/Tmp.java
        subp.call(['javac', 'temp/' + self.tmp_fn + '.java'])

    def remove_test_files(self):
        # Remove the temporary test directory:
        # -> rm -rf temp
        subp.call(['rm', '-rf', 'temp'])

    def make_pred_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def make_pred_in_java(self, features):
        # -> java -classpath temp <temp_filename> <features>
        cmd = ['java', '-classpath', 'temp', self.tmp_fn]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
