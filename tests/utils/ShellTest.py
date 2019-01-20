# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn_porter.utils.Shell import Shell


class ShellTest(TestCase):

    def test_outputs(self):
        out = Shell.check_output('echo 0')
        self.assertEqual(out, '0')

        out = Shell.check_output('echo 1')
        self.assertEqual(out, '1')

        out = Shell.check_output(['echo', 'xyz'], shell=False)
        self.assertEqual(out, 'xyz')

    def test_exceptions(self):
        self.assertRaises(AttributeError, lambda: Shell.check_output([]))
        self.assertRaises(AttributeError, lambda: Shell.check_output(''))
