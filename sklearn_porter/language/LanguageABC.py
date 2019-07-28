# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List


class LanguageABC(ABC):
    @property
    @abstractmethod
    def KEY(self) -> str:
        pass

    @property
    @abstractmethod
    def LABEL(self) -> str:
        pass

    @property
    @abstractmethod
    def DEPENDENCIES(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def TEMP_DIR(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def SUFFIX(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def CMD_COMPILE(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def CMD_EXECUTE(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def TEMPLATES(self) -> Dict[str, str]:
        pass
