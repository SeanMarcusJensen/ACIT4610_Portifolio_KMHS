import logging
from .formatter.logging_formatter import CustomFormatter
from .abstraction import ILogger
from .loggers import BaseLogger, NoLogger
from abc import ABC, abstractmethod


class LoggerFactory(ABC):
    @abstractmethod
    def get_logger(self, name: str) -> ILogger:
        pass


class BaseLoggerFactory(LoggerFactory):
    _instance = None  # Class-level variable to hold singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerFactory, cls).__new__(cls)
        return cls._instance

    def get_logger(self, name: str) -> ILogger:
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setFormatter(CustomFormatter())
            logger.addHandler(ch)
        return BaseLogger(logger)


class NoLoggerFactory(LoggerFactory):
    def get_logger(self, name: str) -> ILogger:
        return NoLogger()
