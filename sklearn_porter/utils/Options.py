# -*- coding: utf-8 -*-

from typing import Callable, Optional


class Options:

    options = {}

    class Option:
        def __init__(
                self,
                name: str,
                value,
                callback: Optional[Callable] = None
        ):
            self.name = name
            self.value = value
            self.callback = callback

    @staticmethod
    def add_option(name: str, callback: Optional[Callable] = None):
        Options.options[name] = Options.Option(name, None, callback)

    @staticmethod
    def set_option(name: str, value):
        if name in Options.options.keys():
            Options.options[name].value = value

            # Apply callback:
            if isinstance(Options.options[name].callback, Callable):
                Options.options[name].callback(Options.options[name].value)

    @staticmethod
    def get_option(key: str, default=None):
        option = Options.options.get(key)
        if not option:
            return default
        return option.value


def set_option(name: str, value):
    Options.set_option(**locals())
