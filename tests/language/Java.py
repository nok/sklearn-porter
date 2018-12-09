# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

import os

from sklearn_porter import Porter
from sklearn_porter.utils.Environment import Environment
from sklearn_porter.utils.Shell import Shell


class Java(object):

    LANGUAGE = 'java'

    # noinspection PyPep8Naming
    def setUp(self):
        Environment.check_deps(['mkdir', 'rm', 'java', 'javac'])
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_estimator(self, export_data=False, embed_data=False):
        self.estimator.fit(self.X, self.y)
        Shell.call('rm -rf tmp')
        Shell.call('mkdir tmp')
        filename = self.tmp_fn + '.java'
        path = os.path.join('tmp', filename)
        with open(path, 'w') as f:
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
        if export_data:
            cmd = 'javac -cp ./gson.jar {}'.format(path)
            Shell.call(cmd)
        else:
            cmd = 'javac ' + path
            Shell.call(cmd)

    def pred_in_py(self, features, cast=True):
        pred = self.estimator.predict([features])[0]
        return int(pred) if cast else float(pred)

    def pred_in_custom(self, features, cast=True, export_data=False):
        if export_data:
            cmd = 'java -cp ./gson.jar:./tmp {}'.format(self.tmp_fn).split()
            cmd += ['./tmp/data.json']
        else:
            cmd = 'java -classpath tmp {}'.format(self.tmp_fn).split()
        args = [str(f).strip() for f in features]
        cmd += args
        pred = Shell.check_output(cmd)
        return int(pred) if cast else float(pred)
