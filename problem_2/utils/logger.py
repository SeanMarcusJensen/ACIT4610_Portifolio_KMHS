from abc import ABC, abstractmethod

class Logger(ABC):
    """Abstract base class for logging functionality."""
    @abstractmethod
    def info(self, **kwargs) -> None:
        pass

    @abstractmethod
    def log(self, **kwargs) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass