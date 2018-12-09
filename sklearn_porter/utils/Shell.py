# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import six

from subprocess import call
from subprocess import check_output
from subprocess import STDOUT


class Shell(object):

    @staticmethod
    def _run(method, cmd, cwd=None, shell=True, universal_newlines=True,
             stderr=STDOUT):
        """Internal wrapper for `call` amd `check_output`"""
        if not cmd:
            error_msg = 'Passed empty text or list'
            raise AttributeError(error_msg)
        if isinstance(cmd, six.string_types):
            cmd = str(cmd)
        if shell:
            if isinstance(cmd, list):
                cmd = ' '.join(cmd)
        else:
            if isinstance(cmd, str):
                cmd = cmd.strip().split()
        out = method(cmd, shell=shell, cwd=cwd, stderr=stderr,
                     universal_newlines=universal_newlines)
        if isinstance(out, bytes):
            out = out.decode('utf-8')
        return str(out).strip()

    @staticmethod
    def call(cmd, shell=True, cwd=None, universal_newlines=True, stderr=STDOUT):
        """Just execute a specific command."""
        return Shell._run(call, cmd, shell=shell, cwd=cwd, stderr=stderr,
                          universal_newlines=universal_newlines)

    @staticmethod
    def check_output(cmd, shell=True, cwd=None, universal_newlines=True,
                     stderr=STDOUT):
        """Execute a specific command and return the output."""
        return Shell._run(check_output, cmd, shell=shell, cwd=cwd,
                          stderr=stderr, universal_newlines=universal_newlines)
