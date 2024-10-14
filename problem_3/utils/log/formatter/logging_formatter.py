import logging
from datetime import datetime


class LogColors:
    RESET = "\033[0m"
    INFO = "\033[32m"    # Green
    DEBUG = "\033[34m"   # Blue
    WARN = "\033[33m"    # Yellow
    ERROR = "\033[31m"   # Red
    CRITICAL = "\033[41m"  # Red background


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Extract logger name from the record
        logger_name = record.name

        # Use current time for the log entry
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Apply color based on log level
        if record.levelname == 'INFO':
            color = LogColors.INFO
        elif record.levelname == 'DEBUG':
            color = LogColors.DEBUG
        elif record.levelname == 'WARNING':
            color = LogColors.WARN
        elif record.levelname == 'ERROR':
            color = LogColors.ERROR
        elif record.levelname == 'CRITICAL':
            color = LogColors.CRITICAL
        else:
            color = LogColors.RESET

        # Format the log message
        log_msg = f"{color}[{logger_name}({record.lineno})][{log_time}][{record.levelname}] {record.getMessage()} {LogColors.RESET}"
        return log_msg
