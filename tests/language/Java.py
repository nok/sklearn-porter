# -*- coding: utf-8 -*-

import os
import subprocess as subp
from sklearn_porter import Porter
from tests.utils.DependencyChecker import DependencyChecker as Checker


class Java(Checker):

    LANGUAGE = 'java'
    DEPENDENCIES = ['mkdir', 'rm', 'java', 'javac']

    # noinspection PyPep8Naming
    def setUp(self):
        super(Java, self).setUp()
        self._check_test_dependencies()
        self._init_test()

    def _init_test(self):
        self.tmp_fn = 'Brain'

    def _port_estimator(self, export_data=False, embed_data=False):
        self.estimator.fit(self.X, self.y)
        subp.call('rm -rf tmp'.split())
        subp.call('mkdir tmp'.split())
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
            cmd = 'javac -cp ./gson.jar {}'.format(path).split()
            subp.call(cmd)
        else:
            subp.call(['javac', path])

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
        pred = subp.check_output(cmd, stderr=subp.STDOUT).rstrip()
        return int(pred) if cast else float(pred)
