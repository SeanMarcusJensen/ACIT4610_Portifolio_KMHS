from ..abstraction import ILogger
from logging import Logger


class NoLogger(ILogger):
    def debug(self, message) -> None:
        pass

    def info(self, message) -> None:
        pass

    def warn(self, message) -> None:
        pass

    def error(self, message) -> None:
        pass

    def critical(self, message) -> None:
        """ Log a message with severity 'CRITICAL'
        TODO: Should maybe raise an exception here.
        """
        pass
