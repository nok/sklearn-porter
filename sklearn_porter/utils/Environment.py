# -*- coding: utf-8 -*-

import os
import sys

try:
    from shutil import which as _which
except ImportError:
    def _which(cmd, mode=os.F_OK | os.X_OK, path=None):
        """Given a command, mode, and a PATH string, return the path which
        conforms to the given mode on the PATH, or None if there is no such
        file.
        `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
        of os.environ.get("PATH"), or can be overridden with a custom search
        path.
        Note: This function was backported from the Python 3 source code.

        # __author__ = "Daniel Roy Greenfeld"
        # __email__ = "pydanny@gmail.com"
        # __version__ = "0.5.3"
        # https://github.com/pydanny/whichcraft/blob/master/whichcraft.py
        """
        def _access_check(fn, mode):
            return os.path.exists(fn) and os.access(fn, mode)\
                   and not os.path.isdir(fn)
        if os.path.dirname(cmd):
            if _access_check(cmd, mode):
                return cmd
            return None
        if path is None:
            path = os.environ.get('PATH', os.defpath)
        if not path:
            return None
        path = path.split(os.pathsep)
        if sys.platform == 'win32':
            if os.curdir not in path:
                path.insert(0, os.curdir)
            pathext = os.environ.get('PATHEXT', '').split(os.pathsep)
            if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
                files = [cmd]
            else:
                files = [cmd + ext for ext in pathext]
        else:
            files = [cmd]
        seen = set()
        for dir in path:
            normdir = os.path.normcase(dir)
            if normdir not in seen:
                seen.add(normdir)
                for thefile in files:
                    name = os.path.join(dir, thefile)
                    if _access_check(name, mode):
                        return name
        return None


class Environment(object):
    """Get information from the system and local environment."""

    @staticmethod
    def read_python_version():
        """Return the local Python version."""
        return sys.version_info[:3]

    @staticmethod
    def read_sklearn_version():
        """Return the local scikit-learn version."""
        from sklearn import __version__ as sklearn_ver
        sklearn_ver = str(sklearn_ver).split('.')
        sklearn_ver = [int(v) for v in sklearn_ver]
        major, minor = sklearn_ver[0], sklearn_ver[1]
        patch = sklearn_ver[2] if len(sklearn_ver) >= 3 else 0
        return major, minor, patch

    @staticmethod
    def read_platform():
        """Return the current system platform."""
        return sys.platform

    @staticmethod
    def check_windows(system=None, raise_exception=True):
        """Is Windows the current using operating system?"""
        if not system:
            system = Environment.read_platform()
        is_win = system in ('cygwin', 'win32', 'win64')
        if is_win and raise_exception:
            error_msg = "Windows isn't supported yet."
            raise OSError(error_msg)
        return is_win

    @staticmethod
    def has_app(name, check_win=True):
        """Check whether the application <name> is installed."""
        if check_win:
            Environment.check_windows()
        return _which(str(name)) is not None

    @staticmethod
    def has_apps(names, check_win=True):
        """Check whether the applications [<name>, ...] are installed."""
        if check_win:
            Environment.check_windows()
        for name in names:
            yield Environment.has_app(str(name), check_win=False)

    @staticmethod
    def check_deps(deps, check_win=True):
        """Check whether specific requirements are installed."""
        if check_win:
            Environment.check_windows()
        if not isinstance(deps, list):
            deps = [deps]
        checks = list(Environment.has_apps(deps, check_win=False))
        if not all(checks):
            for name, available in list(dict(zip(deps, checks)).items()):
                if not available:
                    error_msg = "The required application/dependency '{0}'" \
                                " isn't installed.".format(name)
                    raise EnvironmentError(error_msg)
