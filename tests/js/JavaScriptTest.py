# -*- coding: utf-8 -*-

import random
import time
import subprocess as subp
import os
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils import shuffle

from sklearn_porter import Porter


class JavaScriptTest():

    LANGUAGE = 'js'
    TEST_DEPENDENCIES = ['mkdir', 'rm', 'node']

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
            cmd = 'if hash {} 2/dev/null; then ' \
                  'echo 1; else echo 0; fi'.format(dep)
            available = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                error = "The required test dependency '{0}'" \
                        " is not available.".format(dep)
                self.fail(error)

    def _init_test(self):
        self.tmp_fn = os.path.join('tmp', 'brain.js')
        self.n_random_tests = 150
        if 'N_RANDOM_TESTS' in set(os.environ):
            n_tests = os.environ.get('N_RANDOM_TESTS')
            if str(n_tests).strip().isdigit():
                n_tests = int(n_tests)
                if n_tests > 0:
                    self.n_random_tests = n_tests

    def _init_data(self):
        data = load_iris()
        self.X = shuffle(data.data, random_state=0)
        self.y = shuffle(data.target, random_state=0)
        self.n_features = len(self.X[0])

    def _port_model(self, clf):
        self._clear_model()
        self.clf = clf
        self.clf.fit(self.X, self.y)
        # $ mkdir temp
        subp.call(['mkdir', 'tmp'])
        with open(self.tmp_fn, 'w') as f:
            f.write(Porter(self.clf, language='js').export(
                class_name='Brain', method_name='foo'))
        self._start_test()

    def _clear_model(self):
        self.clf = None
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'tmp'])

    def test_random_features(self):
        Y, Y_py = [], []
        min_vals = np.amin(self.X, axis=0)
        max_vals = np.amax(self.X, axis=0)
        for n in range(self.n_random_tests):
            x = [random.uniform(min_vals[f], max_vals[f]) for f in
                 range(self.n_features)]
            Y.append(self.make_pred_in_custom(x))
            Y_py.append(self.make_pred_in_py(x))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(Y, Y_py)

    def test_existing_features(self):
        Y, Y_py = [], []
        for X in self.X:
            Y.append(self.make_pred_in_custom(X))
            Y_py.append(self.make_pred_in_py(X))
        # noinspection PyUnresolvedReferences
        self.assertListEqual(Y, Y_py)

    def make_pred_in_py(self, features):
        return int(self.clf.predict([features])[0])

    def make_pred_in_custom(self, features):
        # $ node temp/tmp.js <features>
        cmd = ['node', self.tmp_fn]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT)
        return int(pred)
