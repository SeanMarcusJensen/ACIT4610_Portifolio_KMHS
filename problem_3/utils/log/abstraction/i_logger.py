from abc import ABC, abstractmethod


class ILogger(ABC):

    @abstractmethod
    def debug(self, message) -> None:
        pass

    @abstractmethod
    def info(self, message) -> None:
        pass

    @abstractmethod
    def warn(self, message) -> None:
        pass

    @abstractmethod
    def error(self, message) -> None:
        pass

    @abstractmethod
    def critical(self, message) -> None:
        pass
