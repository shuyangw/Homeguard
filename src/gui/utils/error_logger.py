r"""
Error logging utility for GUI application.

Logs all errors, warnings, and info messages to:
C:\Users\qwqw1\Dropbox\cs\stonk\homeguard_gui_logs
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional


# Log directory
LOG_DIR = Path(r"C:\Users\qwqw1\Dropbox\cs\stonk\homeguard_gui_logs")

# Create logger instance
_gui_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """Get or create the GUI logger instance."""
    global _gui_logger

    if _gui_logger is not None:
        return _gui_logger

    # Create log directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    _gui_logger = logging.getLogger("homeguard_gui")
    _gui_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if _gui_logger.handlers:
        return _gui_logger

    # File handler - rotating log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"gui_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Console handler - only show warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    _gui_logger.addHandler(file_handler)
    _gui_logger.addHandler(console_handler)

    # Log startup
    _gui_logger.info("=" * 80)
    _gui_logger.info("GUI Logger initialized")
    _gui_logger.info(f"Log file: {log_file}")
    _gui_logger.info("=" * 80)

    return _gui_logger


def log_error(message: str, exception: Optional[Exception] = None):
    """
    Log an error message.

    Args:
        message: Error message to log
        exception: Optional exception object to include full traceback
    """
    logger = _get_logger()

    if exception:
        # Include full traceback
        tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        logger.error(f"{message}\n{tb_str}")
    else:
        logger.error(message)


def log_warning(message: str):
    """
    Log a warning message.

    Args:
        message: Warning message to log
    """
    logger = _get_logger()
    logger.warning(message)


def log_info(message: str):
    """
    Log an info message.

    Args:
        message: Info message to log
    """
    logger = _get_logger()
    logger.info(message)


def log_debug(message: str):
    """
    Log a debug message.

    Args:
        message: Debug message to log
    """
    logger = _get_logger()
    logger.debug(message)


def log_exception(exception: Exception, context: str = ""):
    """
    Log an exception with full traceback.

    Args:
        exception: Exception to log
        context: Optional context string (e.g., "During backtest execution")
    """
    logger = _get_logger()

    if context:
        message = f"{context}: {str(exception)}"
    else:
        message = str(exception)

    tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    logger.error(f"{message}\n{tb_str}")


def get_log_directory() -> Path:
    """Get the log directory path."""
    return LOG_DIR


def get_latest_log_file() -> Optional[Path]:
    """
    Get the most recent log file.

    Returns:
        Path to latest log file, or None if no logs exist
    """
    if not LOG_DIR.exists():
        return None

    log_files = list(LOG_DIR.glob("gui_*.log"))
    if not log_files:
        return None

    # Return most recent
    return max(log_files, key=lambda p: p.stat().st_mtime)


# Initialize logger on import
_get_logger()
