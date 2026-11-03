# scikit-learn
from sklearn.exceptions import NotFittedError

from sklearn_porter import enums as enum


class PorterError(Exception):
    """The base class of all errors of sklearn-porter."""


class NotFittedEstimatorError(PorterError, NotFittedError):
    def __init__(self, message: str):
        self.message = (
            f'The passed estimator of kind `{message}` is not fitted.'
        )
        super().__init__(self.message)


class NotSupportedYetError(PorterError):
    def __init__(self, message: str):
        hint = (
            'You can check the documentation '
            'or create a new feature request at '
            'https://github.com/nok/sklearn-porter .'
        )
        self.message = message + '\n\n' + hint
        super().__init__(self.message)


class CompilationFailed(PorterError, RuntimeError):
    def __init__(self, message: str):
        self.message = f'Compilation failed:\n\n{message}'
        super().__init__(self.message)


class ExecutionFailed(PorterError, RuntimeError):
    def __init__(self, message: str):
        self.message = f'Execution failed:\n\n{message}'
        super().__init__(self.message)


class MissingDependencyError(PorterError, RuntimeError):
    """A compiler or an interpreter (e.g. the .NET SDK) is missing."""


class CodeTooLarge(CompilationFailed):
    def __init__(self, message: str):
        hint = (
            'Please try to save the model data separately '
            'by changing the template type to `exported`: '
            "`template='exported'`."
        )
        # `CompilationFailed` adds the prefix of the message:
        super().__init__(f'{message}\n\n{hint}')


class TooManyConstants(CompilationFailed):
    def __init__(self, message: str):
        hint = (
            'Please try to save the model data separately '
            'by changing the template type to `exported`: '
            "`template='exported'`."
        )
        # `CompilationFailed` adds the prefix of the message:
        super().__init__(f'{message}\n\n{hint}')


class InvalidMethodError(PorterError, ValueError):
    def __init__(self, message: str):
        opts = ', '.join([f'`{m.value}`' for m in list(enum.Method)])
        self.message = (
            f'The passed method `{message}` is invalid. '
            f'Valid methods are: {opts}.'
        )
        super().__init__(self.message)


class InvalidLanguageError(PorterError, ValueError):
    def __init__(self, message: str):
        opts = ', '.join(
            [f'`{lang.value.KEY}`' for lang in list(enum.Language)]
        )
        self.message = (
            f'The passed language `{message}` is invalid. '
            f'Valid languages are: {opts}.'
        )
        super().__init__(self.message)


class InvalidTemplateError(PorterError, ValueError):
    def __init__(self, message: str):
        opts = ', '.join([f'`{t.value}`' for t in list(enum.Template)])
        self.message = (
            f'The passed template `{message}` is invalid. '
            f'Valid templates are: {opts}.'
        )
        super().__init__(self.message)
