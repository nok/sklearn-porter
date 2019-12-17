# -*- coding: utf-8 -*-

from sklearn_porter import enums as enum


class QualityWarning(Warning):
    pass


class NotFittedEstimatorError(Exception):
    def __init__(self, message: str):
        self.message = 'The passed estimator of kind `{}` ' \
                       'is not fitted.'.format(message)
        super().__init__(self.message)


class NotSupportedYetError(Exception):
    def __init__(self, message: str):
        hint = 'You can check the documentation ' \
               'or create a new feature request at ' \
               'https://github.com/nok/sklearn-porter .'
        self.message = message + '\n\n' + hint
        super().__init__(self.message)


class CompilationFailed(RuntimeError):
    def __init__(self, message: str):
        self.message = 'Compilation failed:\n\n{}'.format(message)
        super().__init__(self.message)


class CodeTooLarge(CompilationFailed):
    def __init__(self, message: str):
        hint = 'Please try to save the model data separately ' \
               'by changing the template type to `exported`: ' \
               '`template=\'exported\'`.'
        self.message = 'Compilation failed:\n\n{}\n\n{}'.format(message, hint)
        super().__init__(self.message)


class TooManyConstants(CompilationFailed):
    def __init__(self, message: str):
        hint = 'Please try to save the model data separately ' \
               'by changing the template type to `exported`: ' \
               '`template=\'exported\'`.'
        self.message = 'Compilation failed:\n\n{}\n\n{}'.format(message, hint)
        super().__init__(self.message)


class InvalidMethodError(Exception):
    def __init__(self, message: str):
        opts = ', '.join(['`{}`'.format(m.value) for m in list(enum.Method)])
        self.message = 'The passed method `{}` is invalid. ' \
                       'Valid methods are: {}.'.format(message, opts)
        super().__init__(self.message)


class InvalidLanguageError(Exception):
    def __init__(self, message: str):
        opts = ', '.join(
            ['`{}`'.format(l.value.KEY) for l in list(enum.Language)]
        )
        self.message = 'The passed language `{}` is invalid. ' \
                       'Valid languages are: {}.'.format(message, opts)
        super().__init__(self.message)


class InvalidTemplateError(Exception):
    def __init__(self, message: str):
        opts = ', '.join(['`{}`'.format(t.value) for t in list(enum.Template)])
        self.message = 'The passed template `{}` is invalid. ' \
                       'Valid templates are: {}.'.format(message, opts)
