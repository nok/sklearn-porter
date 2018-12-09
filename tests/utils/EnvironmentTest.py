# -*- coding: utf-8 -*-

from unittest import TestCase

from sklearn_porter.utils.Environment import Environment


class EnvironmentTest(TestCase):

    def test_is_windows_darwin(self):
        self.assertFalse(Environment._platform_is_windows('darwin'))

    def test_is_windows_win32(self):
        self.assertRaises(OSError, lambda: Environment._platform_is_windows('win32'))

    def test_has_application_true(self):
        has_app = Environment.has_app('ls', check_platform=False)
        self.assertTrue(has_app)

    def test_has_application_false(self):
        has_app = Environment.has_app('m0dn4r', check_platform=False)
        self.assertFalse(has_app)

    def test_has_applications_true(self):
        has_apps = Environment.has_apps(['ls', 'which'], check_platform=False)
        has_apps = all(list(has_apps))
        self.assertTrue(has_apps)

    def test_has_applications_false(self):
        has_apps = Environment.has_apps(['ls', 'm0dn4r'], check_platform=False)
        has_apps = all(list(has_apps))
        self.assertFalse(has_apps)
