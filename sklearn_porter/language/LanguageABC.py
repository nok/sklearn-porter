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
    # The command to compile the generated source code (if required):
    CMD_COMPILE: ClassVar[str | None]
    # The command to execute (the compiled) source code:
    CMD_EXECUTE: ClassVar[str]
    # The basic templates of the programming language:
    TEMPLATES: ClassVar[dict[str, str]]
