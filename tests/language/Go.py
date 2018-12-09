# -*- coding: utf-8 -*-

import os

from sklearn_porter import Porter
from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell


class Go(object):

    LANGUAGE = 'go'

    # noinspection PyPep8Naming
    def setUp(self):
        Environment.check_deps(['mkdir', 'rm', 'go'])
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'brain'

    def _port_estimator(self):
        self.estimator.fit(self.X, self.y)
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        path = os.path.join('.', 'tmp', self.tmp_fn + '.go')
        output = os.path.join('.', 'tmp', self.tmp_fn)
        with open(path, 'w') as f:
            porter = Porter(self.estimator, language=self.LANGUAGE)
            out = porter.export(class_name='Brain', method_name='foo')
            f.write(out)
        cmd = 'go build -o {} {}'.format(output, path)
        Shell.call(cmd)

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True):
        # $ ./<temp_filename> <features>
        cmd = [os.path.join('.', 'tmp', self.tmp_fn)]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = Shell.check_output(cmd)
        return int(pred) if cast else float(pred)
