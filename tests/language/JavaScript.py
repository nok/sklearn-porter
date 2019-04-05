# -*- coding: utf-8 -*-

import os

from sklearn_porter import Porter
from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell


class JavaScript(object):

    LANGUAGE = 'js'

    # noinspection PyPep8Naming
    def setUp(self):
        Environment.check_deps(['mkdir', 'rm', 'node'])
        self._init_test()

    def _init_test(self):
        self.tmp_fn = os.path.join('tmp', 'Brain.js')

    def _port_estimator(self, export_data=False, embed_data=False):
        self.estimator.fit(self.X, self.y)
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        with open(self.tmp_fn, 'w') as f:
            porter = Porter(self.estimator, language=self.LANGUAGE)
            if export_data:
                out = porter.export(class_name='Brain',
                                    method_name='foo',
                                    export_data=True,
                                    export_dir='tmp')
            else:
                out = porter.export(class_name='Brain',
                                    method_name='foo',
                                    embed_data=embed_data)
            f.write(out)

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True, export_data=False):
        cmd = ['node', self.tmp_fn]
        if export_data:
            cmd += ['http://0.0.0.0:8713/tmp/data.json']
        args = [str(f).strip() for f in features]
        cmd += args
        pred = Shell.check_output(cmd)
        return int(pred) if cast else float(pred)
