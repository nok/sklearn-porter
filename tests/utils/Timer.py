# -*- coding: utf-8 -*-

import sys
import time


class Timer(object):

    def _start_test(self):
        self.start_time = time.time()

    def _stop_test(self):
        delta = time.time() - self.start_time
        sys.stdout.write('(%.3f sec) ' % delta)
        sys.stdout.flush()
