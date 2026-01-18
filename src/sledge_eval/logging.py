"""Logging utilities for sledge-eval."""

import logging
import sys
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"


def get_logger(
    name: str,
    level: int = logging.INFO,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger for sledge-eval modules.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        debug: If True, use DEBUG level and detailed format
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set level based on debug flag
    if debug:
        logger.setLevel(logging.DEBUG)
        log_format = DEBUG_FORMAT
    else:
        logger.setLevel(level)
        log_format = DEFAULT_FORMAT

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure the root logger for the application.

    Args:
        debug: If True, use DEBUG level
        log_file: Optional file path to write logs to
    """
    level = logging.DEBUG if debug else logging.INFO
    log_format = DEBUG_FORMAT if debug else DEFAULT_FORMAT

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
    )


class LoggerMixin:
    """Mixin class to provide logging capabilities to evaluator classes."""

    _logger: Optional[logging.Logger] = None

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        if self._logger is None:
            self._logger = get_logger(
                self.__class__.__module__,
                debug=getattr(self, "debug", False),
            )
        return self._logger

    def log_debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def log_info(self, message: str, *args, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)

    def log_exception(self, message: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
