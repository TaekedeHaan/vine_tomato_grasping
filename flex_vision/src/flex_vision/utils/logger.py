"""
    file: logger.py
    description: This module contains the required functionality to create a logger
    author: Taeke de Haan
    date: 28-11-2022
"""
import logging
import os
import typing


def init_logger(name="", log_to_screen=True, logger_file=None, level=logging.DEBUG):
    # type: (str, bool, typing.Optional[str], int) -> logging.Logger
    """Initialize a new logger or get existing logger. Compatible with ROS

    Args:
        name (string, optional): Name of the logger. Defaults to "".
        log_to_screen (bool, optional): If true will log to screen. Defaults to True.
        logger_file (string, optional): file to store the log in. Log will not be stored if left empty. Defaults to None.
        level (int, optional): The logging level. Defaults to logging.DEBUG.

    Returns:
        Logger: The logger
    """
    my_logger = logging.getLogger(name)

    format_log = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
    format_date_time = "%Y-%b-%d %H:%M:%S"
    console_formatter = CustomFormatter(format_log, format_date_time)
    file_formatter = logging.Formatter(format_log, format_date_time)
    my_logger.setLevel(level)

    if log_to_screen:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        my_logger.addHandler(console_handler)

    if logger_file:
        log_directory = os.path.dirname(logger_file)

        # create dirs if required
        if not os.path.isdir(log_directory):
            os.makedirs(log_directory)

        file_handler = logging.FileHandler(logger_file)
        file_handler.setFormatter(file_formatter)
        my_logger.addHandler(file_handler)

    return my_logger


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    COLOR_DEBUG = "\033[32m"
    COLOR_INFO = "\033[0m"
    COLOR_WARNING = "\033[33m"
    COLOR_ERROR = "\033[31;1m"
    COLOR_CRITICAL = "\033[31m"
    DEFAULT_COLOR = '\x1b[0m'

    def __init__(self, fmt=None, datefmt=None):
        # type: (typing.Optional[str], typing.Optional[str]) -> None
        super(CustomFormatter, self).__init__(fmt=fmt, datefmt=datefmt)
        self._fmt = typing.cast(str, self._fmt)  # typing does not understand that _fmt is of type str
        self.FORMATS = {
            logging.DEBUG: self.COLOR_DEBUG + self._fmt + self.DEFAULT_COLOR,
            logging.INFO: self.COLOR_INFO + self._fmt + self.DEFAULT_COLOR,
            logging.WARNING: self.COLOR_WARNING + self._fmt + self.DEFAULT_COLOR,
            logging.ERROR: self.COLOR_ERROR + self._fmt + self.DEFAULT_COLOR,
            logging.CRITICAL: self.COLOR_CRITICAL + self._fmt + self.DEFAULT_COLOR
        }

    def format(self, record):
        # type: (logging.LogRecord) -> str
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)
