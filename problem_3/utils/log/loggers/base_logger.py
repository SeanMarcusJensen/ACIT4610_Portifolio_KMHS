from ..abstraction import ILogger
from logging import Logger


class BaseLogger(ILogger):
    def __init__(self, logger: Logger):
        self.logger = logger

    def debug(self, message) -> None:
        self.logger.debug(message)

    def info(self, message) -> None:
        self.logger.info(message)

    def warn(self, message) -> None:
        self.logger.warning(message)

    def error(self, message) -> None:
        self.logger.error(message)

    def critical(self, message) -> None:
        """ Log a message with severity 'CRITICAL'
        TODO: Should maybe raise an exception here.
        """
        self.logger.critical(message)
