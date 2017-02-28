# -*- coding: utf-8 -*-

import random
import time
import subprocess as subp
import os
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils import shuffle


class CTest():

    LANGUAGE = 'c'
    TEST_DEPENDENCIES = ['mkdir', 'gcc']

    # noinspection PyPep8Naming
    def setUp(self):
        self._check_test_dependencies()
        self._init_test()
        self._init_data()

    # noinspection PyPep8Naming
    def tearDown(self):
        self._stop_test()
        self._clear_model()

    def _start_test(self):
        self.start_time = time.time()

    def _stop_test(self):
        print('%.3fs' % (time.time() - self.start_time))

    def _check_test_dependencies(self):
        for dep in self.TEST_DEPENDENCIES:
            # $ if hash gcc 2/dev/null; then echo 1; else echo 0; fi
            cmd = 'if hash %s 2/dev/null; then echo 1; else echo 0; fi' % dep
            available = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                err_msg = ('The required test dependency \'{0}\' '
                           'is not available.').format(dep)
                self.fail(err_msg)

    def _init_test(self):
        self.tmp_fn = 'tmp'
        self.n_random_tests = 150
        if 'N_RANDOM_TESTS' in set(os.environ):
            n = os.environ.get('N_RANDOM_TESTS')
            if str(n).strip().isdigit():
                if int(n) > 0:
                    self.n_random_tests = int(n)

    def _init_data(self):
        data = load_iris()
        self.X = shuffle(data.data, random_state=0)
        self.y = shuffle(data.target, random_state=0)
        self.n_features = len(self.X[0])

    def _port_model(self, clf):
        self._clear_model()
        self.clf = clf
        self.clf.fit(self.X, self.y)
        subp.call(['mkdir', 'temp'])  # $ mkdir temp
        # Save transpiled model:
        filename = self.tmp_fn + '.c'
        path = os.path.join('temp', filename)
        with open(path, 'w') as f:
            f.write(self.porter.port(self.clf))
        # $ gcc temp/tmp.c -o temp/tmp
        subp.call(['gcc', path, '-lm', '-o', 'temp/' + self.tmp_fn])
        self._start_test()

    def _clear_model(self):
        self.clf = None
        subp.call(['rm', '-rf', 'temp'])  # $ rm -rf temp

    def test_random_features(self):
        java_preds, py_preds = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.n_random_tests):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            java_preds.append(self.make_pred_in_c(x))
            py_preds.append(self.make_pred_in_py(x))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(py_preds, java_preds)

    def test_existing_features(self):
        java_preds, py_preds = [], []
        for X in self.X:
            java_preds.append(self.make_pred_in_c(X))
            py_preds.append(self.make_pred_in_py(X))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(java_preds, py_preds)

    def make_pred_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def make_pred_in_c(self, features):
        cmd = [os.path.join('.', 'temp', 'tmp')]  # $ <temp_filename> <features>
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
