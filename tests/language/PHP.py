# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

import os

from sklearn_porter import Porter
from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell


class PHP(object):

    LANGUAGE = 'php'

    # noinspection PyPep8Naming
    def setUp(self):
        Environment.check_deps(['mkdir', 'rm', 'php'])
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_estimator(self):
        self.estimator.fit(self.X, self.y)
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        with open(path, 'w') as f:
            porter = Porter(self.estimator, language=self.LANGUAGE)
            out = porter.export(class_name='Brain', method_name='foo')
            f.write(out)

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True):
        # $ php -f <tmp_filename> <features>
        filename = self.tmp_fn + '.php'
        path = os.path.join('tmp', filename)
        cmd = 'php -f {}'.format(path).split()
        args = [str(f).strip() for f in features]
        cmd += args
        pred = Shell.check_output(cmd)
        return int(pred) if cast else float(pred)
