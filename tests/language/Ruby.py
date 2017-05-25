# -*- coding: utf-8 -*-

import os
import subprocess as subp
from sklearn_porter import Porter
from ..utils.DependencyChecker import DependencyChecker as Checker


class Ruby(Checker):

    LANGUAGE = 'ruby'
    DEPENDENCIES = ['mkdir', 'rm', 'ruby']

    # noinspection PyPep8Naming
    def setUp(self):
        super(Ruby, self).setUp()
        self._check_test_dependencies()
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_model(self):
        self.mdl.fit(self.X, self.y)
        subp.call(['rm', '-rf', 'tmp'])
        subp.call(['mkdir', 'tmp'])
        filename = self.tmp_fn + '.rb'
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
        # $ ruby temp <temp_filename> <features>
        filename = self.tmp_fn + '.rb'
        path = os.path.join('tmp', filename)
        cmd = ['ruby', path]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = subp.check_output(cmd, stderr=subp.STDOUT).rstrip()
        return int(pred) if cast else float(pred)
