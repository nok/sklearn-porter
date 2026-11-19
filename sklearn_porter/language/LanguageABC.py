import re
from abc import ABC
from typing import ClassVar


class LanguageABC(ABC):
    """The basic definitions of a supported programming language."""

    # The abbreviation of the programming language:
    KEY: ClassVar[str]
    # The human-readable full name of the programming language:
    LABEL: ClassVar[str]
    # The applications which are required for the system calls:
    DEPENDENCIES: ClassVar[list[str]]
    # The suffix of the generated source files:
    SUFFIX: ClassVar[str]
    # The command (program and arguments) to compile the generated source
    # code (if required), the arguments can contain placeholders like
    # `{src_path}`:
    CMD_COMPILE: ClassVar[list[str] | None]
    # The command (program and arguments) to execute (the compiled) source
    # code, the features are appended as further arguments:
    CMD_EXECUTE: ClassVar[list[str]]
    # The basic templates of the programming language:
    TEMPLATES: ClassVar[dict[str, str]]

    @staticmethod
    def str_literal(value: str) -> str:
        """
        Get the string literal of a value, e.g. of a class label.

        Parameters
        ----------
        value : str
            The value.

        Returns
        -------
        The string literal with all required escapes.
        """
        raise NotImplementedError

    @classmethod
    def methods(cls, source: str, class_name: str) -> list[dict]:
        """
        Get the public prediction methods (e.g. `predict` and
        `predict_proba`) of a ported estimator, which the class of a
        pipeline forwards.

        Parameters
        ----------
        source : str
            The source code of the ported estimator.
        class_name : str
            The class name of the ported estimator.

        Returns
        -------
        The methods in the order of the source code, as dicts with the
        `name`, the return `type` (if the language has types), whether the
        method is `static` and the `params` (in C).
        """
        raise NotImplementedError


def _indent(line: str) -> str:
    """Get the leading whitespace of a line."""
    return line[: len(line) - len(line.lstrip())]


def class_members(source: str, start: re.Pattern) -> list[str]:
    """
    Get the lines of the members of a class (without nested classes).

    Parameters
    ----------
    source : str
        The source code.
    start : re.Pattern
        The pattern of the first line of the class.

    Returns
    -------
    The lines of the class with the indentation of its members (the least
    indentation in the class), i.e. without the lines of nested classes
    and methods.
    """
    lines = source.split('\n')
    first = next(i for i, line in enumerate(lines) if start.match(line))
    indent = _indent(lines[first])
    body = []
    for line in lines[first + 1 :]:
        if line.strip() and _indent(line) == indent:
            break
        body.append(line)
    member = min((_indent(line) for line in body if line.strip()), key=len)
    return [line for line in body if _indent(line) == member and line.strip()]
