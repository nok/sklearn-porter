# -*- coding: utf-8 -*-


class SklearnPorterError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NotSupportedError(SklearnPorterError):
    def __init__(self, message: str):
        hint = 'You can check the documentation ' \
               'or create a new feature request at ' \
               'https://github.com/nok/sklearn-porter .'
        self.message = message + '\n' + hint
        super().__init__(message)
