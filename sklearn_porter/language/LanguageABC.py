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
