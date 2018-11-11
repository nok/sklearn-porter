# -*- coding: utf-8 -*-

import os
import sys

try:
    from shutil import which
except ImportError:
    def which(cmd, mode=os.F_OK | os.X_OK, path=None):
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

    @staticmethod
    def _platform_is_windows(platform=sys.platform):
        """Is the current OS a Windows?"""
        matched = platform in ('cygwin', 'win32', 'win64')
        if matched:
            error_msg = "Windows isn't supported yet"
            raise OSError(error_msg)
        return matched

    @staticmethod
    def has_app(name, check_platform=True):
        """Check whether the application <name> is installed."""
        if check_platform:
            Environment._platform_is_windows()
        return which(str(name)) is not None

    @staticmethod
    def has_apps(names, check_platform=True):
        """Check whether the applications [<name>, ...] are installed."""
        if check_platform:
            Environment._platform_is_windows()
        for name in names:
            yield Environment.has_app(str(name), check_platform=False)

    @staticmethod
    def check_deps(deps):
        """check whether specific requirements are available."""
        if not isinstance(deps, list):
            deps = [deps]
        checks = list(Environment.has_apps(deps))
        if not all(checks):
            for name, available in list(dict(zip(deps, checks)).items()):
                if not available:
                    error_msg = "The required application/dependency '{0}'" \
                                " isn't available.".format(name)
                    raise SystemError(error_msg)
