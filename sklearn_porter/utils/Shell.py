# -*- coding: utf-8 -*-

import subprocess


class Shell(object):

    @staticmethod
    def _str_to_list(text):
        if isinstance(text, list):
            return text
        if not text:
            error_msg = 'Passed empty text'
            raise AttributeError(error_msg)
        if isinstance(text, str):
            text = text.strip().split()
        if not isinstance(text, list):
            text = list(text)
        return text

    @staticmethod
    def call(cmd, cwd=None):
        cmd = Shell._str_to_list(cmd)
        return subprocess.call(cmd, cwd=cwd)

    @staticmethod
    def check_output(cmd, cwd=None, shell=True, stderr=subprocess.STDOUT):
        cmd = Shell._str_to_list(cmd)
        return subprocess.check_output(cmd, cwd=cwd, stderr=stderr, shell=shell)
