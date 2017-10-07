# -*- coding: utf-8 -*-

import os
import subprocess as subp
from sklearn_porter import Porter
from tests.utils.DependencyChecker import DependencyChecker as Checker


class Go(Checker):

    LANGUAGE = 'go'
    DEPENDENCIES = ['mkdir', 'rm', 'go']

    # noinspection PyPep8Naming
    def setUp(self):
        super(Go, self).setUp()
        self._check_test_dependencies()
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'brain'

    def _port_estimator(self):
        self.estimator.fit(self.X, self.y)
        subp.call('rm -rf tmp'.split())
        subp.call('mkdir tmp'.split())
        path = os.path.join('.', 'tmp', self.tmp_fn + '.go')
        output = os.path.join('.', 'tmp', self.tmp_fn)
        with open(path, 'w') as f:
            porter = Porter(self.estimator, language=self.LANGUAGE)
            out = porter.export(class_name='Brain', method_name='foo')
            f.write(out)
        cmd = 'go build -o {} {}'.format(output, path)
        subp.call(cmd.split())

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True):
        # $ ./<temp_filename> <features>
        cmd = [os.path.join('.', 'tmp', self.tmp_fn)]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT).rstrip()
        return int(pred) if cast else float(pred)
