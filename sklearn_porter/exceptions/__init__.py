from sklearn_porter import enums as enum


class NotFittedEstimatorError(Exception):
    def __init__(self, message: str):
        self.message = (
            f'The passed estimator of kind `{message}` is not fitted.'
        )
        super().__init__(self.message)


class NotSupportedYetError(Exception):
    def __init__(self, message: str):
        hint = (
            'You can check the documentation '
            'or create a new feature request at '
            'https://github.com/nok/sklearn-porter .'
        )
        self.message = message + '\n\n' + hint
        super().__init__(self.message)


class CompilationFailed(RuntimeError):
    def __init__(self, message: str):
        self.message = f'Compilation failed:\n\n{message}'
        super().__init__(self.message)


class ExecutionFailed(RuntimeError):
    def __init__(self, message: str):
        self.message = f'Execution failed:\n\n{message}'
        super().__init__(self.message)


class CodeTooLarge(CompilationFailed):
    def __init__(self, message: str):
        hint = (
            'Please try to save the model data separately '
            'by changing the template type to `exported`: '
            "`template='exported'`."
        )
        self.message = f'Compilation failed:\n\n{message}\n\n{hint}'
        super().__init__(self.message)


class TooManyConstants(CompilationFailed):
    def __init__(self, message: str):
        hint = (
            'Please try to save the model data separately '
            'by changing the template type to `exported`: '
            "`template='exported'`."
        )
        self.message = f'Compilation failed:\n\n{message}\n\n{hint}'
        super().__init__(self.message)


class InvalidMethodError(Exception):
    def __init__(self, message: str):
        opts = ', '.join([f'`{m.value}`' for m in list(enum.Method)])
        self.message = (
            f'The passed method `{message}` is invalid. '
            f'Valid methods are: {opts}.'
        )
        super().__init__(self.message)


class InvalidLanguageError(Exception):
    def __init__(self, message: str):
        opts = ', '.join(
            [f'`{lang.value.KEY}`' for lang in list(enum.Language)]
        )
        self.message = (
            f'The passed language `{message}` is invalid. '
            f'Valid languages are: {opts}.'
        )
        super().__init__(self.message)


class InvalidTemplateError(Exception):
    def __init__(self, message: str):
        opts = ', '.join([f'`{t.value}`' for t in list(enum.Template)])
        self.message = (
            f'The passed template `{message}` is invalid. '
            f'Valid templates are: {opts}.'
        )
