# -*- coding: utf-8 -*-

import subprocess as subp


class DependencyChecker(object):

    def _check_test_dependencies(self):
        for dep in self.DEPENDENCIES:
            cmd = 'if hash {} 2/dev/null; then ' \
                  'echo 1; else echo 0; fi'.format(dep)
            available = subp.check_output(cmd, shell=True, stderr=subp.STDOUT)
            available = available.strip() is '1'
            if not available:
                error = "The required test dependency '{0}'" \
                        " is not available.".format(dep)
                self.fail(error)
