# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import absolute_import

from unittest import TestCase

from sklearn_porter.utils.Shell import Shell


class ShellTest(TestCase):

    def test_check_output_echo_num_0(self):
        self.assertEqual(Shell.check_output('echo 0'), '0')

    def test_check_output_echo_num_1(self):
        self.assertEqual(Shell.check_output('echo 1'), '1')

    def test_check_output_echo_list_xyz(self):
        self.assertEqual(Shell.check_output(['echo', 'xyz'], shell=False), 'xyz')

    def test_check_output_empty_list(self):
        self.assertRaises(AttributeError, lambda: Shell.check_output([]))

    def test_check_output_empty_text(self):
        self.assertRaises(AttributeError, lambda: Shell.check_output(''))
