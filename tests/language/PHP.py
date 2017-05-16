# -*- coding: utf-8 -*-

import os
import subprocess as subp

from sklearn_porter import Porter
from ..utils.Timer import Timer
from ..utils.DependencyChecker import DependencyChecker as Checker


class PHP(Timer, Checker):

    LANGUAGE = 'php'
    N_RANDOM_TESTS = 150
    DEPENDENCIES = ['mkdir', 'rm', 'php']

    # noinspection PyPep8Naming
    def setUp(self):
        super(PHP, self).setUp()
        self._check_test_dependencies()
        self._init_test()

    # noinspection PyPep8Naming
    def tearDown(self):
        self._stop_test()
        self._clear_model()

    def _init_test(self):
        self.tmp_fn = 'Brain'
        if 'N_RANDOM_TESTS' in set(os.environ):
            n_tests = os.environ.get('N_RANDOM_TESTS')
            if str(n_tests).strip().isdigit():
                n_tests = int(n_tests)
                if n_tests > 0:
                    self.N_RANDOM_TESTS = n_tests

    def _port_model(self, mdl):
        self._clear_model()
        self.mdl = mdl
        self.mdl.fit(self.X, self.y)
        # $ mkdir temp
        subp.call(['mkdir', 'tmp'])
        # Save transpiled model:
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        with open(path, 'w') as f:
            porter = Porter(self.mdl, language=self.LANGUAGE)
            out = porter.export(class_name='Brain',
                                method_name='foo')
            f.write(out)
        self._start_test()

    def _clear_model(self):
        self.mdl = None
        # $ rm -rf temp
        subp.call(['rm', '-rf', 'tmp'])

    def make_pred_in_py(self, features, cast=True):
        pred = self.mdl.predict([features])[0]
        return int(pred) if cast else float(pred)

    def make_pred_in_custom(self, features, cast=True):
        # $ php -f <temp_filename> <features>
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        cmd = ['php', '-f', path]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT).rstrip()
        return int(pred) if cast else float(pred)
