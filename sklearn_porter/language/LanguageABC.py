from abc import ABC, abstractmethod


class LanguageABC(ABC):
    @property
    @abstractmethod
    def KEY(self) -> str:
        """The abbreviation of the programming language."""

    @property
    @abstractmethod
    def LABEL(self) -> str:
        """The human-readable full name of the programming language."""

    @property
    @abstractmethod
    def DEPENDENCIES(self) -> list[str]:
        """List of dependencies which are required for the system calls."""

    @property
    @abstractmethod
    def SUFFIX(self) -> list[str]:
        """The suffix of the generated source files."""

    @property
    @abstractmethod
    def CMD_COMPILE(self) -> list[str]:
        """The command to compile the generated source code."""

    @property
    @abstractmethod
    def CMD_EXECUTE(self) -> list[str]:
        """The command to execute (the compiled) source code."""

    @property
    @abstractmethod
    def TEMPLATES(self) -> dict[str, str]:
        """Dictionary of basic templates of the programming language."""
