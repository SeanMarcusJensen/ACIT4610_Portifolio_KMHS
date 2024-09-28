from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def info(self, **kwargs) -> None:
        pass