# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn_porter.utils.Environment import Environment


class EnvironmentTest(TestCase):

    def test_check_windows(self):
        self.assertRaises(OSError, lambda: Environment.check_windows(
            'win32', raise_exception=True))

        test_windows = Environment.check_windows('win32', raise_exception=False)
        self.assertTrue(test_windows)

        test_darwin = Environment.check_windows('darwin', raise_exception=False)
        self.assertFalse(test_darwin)

    def test_has_applications(self):
        has_app = Environment.has_app('ls', check_win=False)
        self.assertTrue(has_app)

        has_app = Environment.has_app('mkdir', check_win=False)
        self.assertTrue(has_app)

        has_app = Environment.has_app('appl1c4710n', check_win=False)
        self.assertFalse(has_app)

        has_apps = Environment.has_apps(['ls', 'which'], check_win=False)
        self.assertTrue(all(list(has_apps)))

        has_apps = Environment.has_apps(['ls', 'appl1c4710n'], check_win=False)
        self.assertFalse(all(list(has_apps)))

    def test_check_dependencies(self):
        self.assertRaises(EnvironmentError, lambda: Environment.check_deps('appl1c4710n', check_win=False))
        self.assertRaises(EnvironmentError, lambda: Environment.check_deps(['appl1c4710n'], check_win=False))
