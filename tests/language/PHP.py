# -*- coding: utf-8 -*-

import os
import subprocess as subp
from sklearn_porter import Porter
from tests.utils.DependencyChecker import DependencyChecker as Checker


class PHP(Checker):

    LANGUAGE = 'php'
    DEPENDENCIES = ['mkdir', 'rm', 'php']

    # noinspection PyPep8Naming
    def setUp(self):
        super(PHP, self).setUp()
        self._check_test_dependencies()
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_model(self):
        self.mdl.fit(self.X, self.y)
        subp.call('rm -rf tmp'.split())
        subp.call('mkdir tmp'.split())
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        with open(path, 'w') as f:
            porter = Porter(self.mdl, language=self.LANGUAGE)
            out = porter.export(class_name='Brain',
                                method_name='foo')
            f.write(out)

    def pred_in_py(self, features, cast=True):
        pred = self.mdl.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True):
        # $ php -f <tmp_filename> <features>
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        cmd = 'php -f {}'.format(path).split()
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT).rstrip()
        return int(pred) if cast else float(pred)
