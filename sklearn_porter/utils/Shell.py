# -*- coding: utf-8 -*-

import subprocess as subp


class Shell(object):

    @staticmethod
    def call(command, cwd=None):
        if isinstance(command, str):
            command = command.split()
        if isinstance(command, list):
            return subp.call(command, cwd=cwd)
        return None

    @staticmethod
    def check_output(command, cwd=None, shell=True, stderr=subp.STDOUT):
        if isinstance(command, str):
            command = command.split()
        if isinstance(command, list):
            subp.check_output(command, shell=shell, cwd=cwd, stderr=stderr)
        return None
