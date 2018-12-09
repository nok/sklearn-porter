# -*- coding: utf-8 -*-

import os

from sklearn_porter import Porter
from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell


class Ruby():

    LANGUAGE = 'ruby'

    # noinspection PyPep8Naming
    def setUp(self):
        Environment.check_deps(['mkdir', 'rm', 'ruby'])
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_estimator(self):
        self.estimator.fit(self.X, self.y)
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        filename = self.tmp_fn + '.rb'
        path = os.path.join('tmp', filename)
        with open(path, 'w') as f:
            porter = Porter(self.estimator, language=self.LANGUAGE)
            out = porter.export(class_name='Brain', method_name='foo')
            f.write(out)

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True):
        # $ ruby temp <temp_filename> <features>
        filename = self.tmp_fn + '.rb'
        path = os.path.join('tmp', filename)
        cmd = ['ruby', path]
        args = [str(f).strip() for f in features]
        cmd += args
        pred = Shell.check_output(cmd)
        return int(pred) if cast else float(pred)
