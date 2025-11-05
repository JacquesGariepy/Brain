"""
Brain Centralized Logging

Provides unified logging interface for the Brain framework.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class BrainLogger:
    """
    Centralized logger for Brain framework.

    Features:
    - Colored console output
    - File logging
    - Multiple log levels
    - Structured logging support
    """

    def __init__(
        self,
        name: str = "brain",
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        console: bool = True,
        file_logging: bool = True,
    ):
        """
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            console: Whether to log to console
            file_logging: Whether to log to file
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        if console:
            self._add_console_handler()

        # File handler
        if file_logging and log_dir:
            self._add_file_handler(log_dir)

    def _add_console_handler(self):
        """Add colored console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Format with colors
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _add_file_handler(self, log_dir: str):
        """Add file handler"""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f"{self.name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(msg, *args, **kwargs)


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def format(self, record):
        """Format log record with colors"""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


# Global logger instance
_global_logger: Optional[BrainLogger] = None


def get_logger(
    name: str = "brain",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
    file_logging: bool = False,
) -> BrainLogger:
    """
    Get or create a Brain logger.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Whether to log to console
        file_logging: Whether to log to file

    Returns:
        BrainLogger instance

    Example:
        >>> logger = get_logger("brain.training")
        >>> logger.info("Starting training...")
        >>> logger.error("Training failed!", exc_info=True)
    """
    global _global_logger

    if _global_logger is None or _global_logger.name != name:
        _global_logger = BrainLogger(
            name=name,
            level=level,
            log_dir=log_dir,
            console=console,
            file_logging=file_logging,
        )

    return _global_logger


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file_logging: bool = False,
):
    """
    Setup global logging configuration.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_dir: Directory for log files
        console: Whether to log to console
        file_logging: Whether to log to file

    Example:
        >>> setup_logging(level="DEBUG", log_dir="./logs", file_logging=True)
        >>> logger = get_logger()
        >>> logger.debug("Debug message")
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.INFO)

    global _global_logger
    _global_logger = BrainLogger(
        name="brain",
        level=log_level,
        log_dir=log_dir,
        console=console,
        file_logging=file_logging,
    )


# Test function
def test_logger():
    """Test logger functionality"""
    print("Testing Brain Logger...")

    # Setup logging
    setup_logging(level="DEBUG", console=True)

    logger = get_logger()

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n✓ Logger tests complete!")


if __name__ == "__main__":
    test_logger()
