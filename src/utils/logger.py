"""
Centralized logging module using Rich for colored console output.

All logging that is intended to be written to disk or displayed to users
must go through this module.
"""

from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.theme import Theme
import sys

# Define custom theme for consistent coloring
custom_theme = Theme({
    "success": "bold green",
    "profit": "bold green",
    "error": "bold red",
    "loss": "bold red",
    "warning": "bold yellow",
    "info": "bold cyan",
    "header": "bold magenta",
    "metric": "bold blue",
    "neutral": "white",
    "dim": "dim white"
})

# Create console instance with custom theme
console = Console(theme=custom_theme, file=sys.stdout)


class Logger:
    """
    Centralized logger using Rich for colored console output.

    Usage:
        logger = Logger()
        logger.success("Trade executed successfully")
        logger.error("Failed to load data")
        logger.info("Loading symbols...")
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize logger.

        Args:
            log_file: Optional file path to write logs to disk
        """
        self.log_file = log_file
        self.console = console

    def _log(self, message: str, style: Optional[str] = None, to_file: bool = True):
        """Internal logging method."""
        # Print to console with color
        if style:
            self.console.print(message, style=style)
        else:
            self.console.print(message)

        # Write to file if specified (without color codes)
        if to_file and self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")

    def success(self, message: str, to_file: bool = True):
        """Log success message (green)."""
        self._log(f"[+] {message}", style="success", to_file=to_file)

    def profit(self, message: str, to_file: bool = True):
        """Log profit/gain message (green)."""
        self._log(f"[^] {message}", style="profit", to_file=to_file)

    def error(self, message: str, to_file: bool = True):
        """Log error message (red)."""
        self._log(f"[X] {message}", style="error", to_file=to_file)

    def loss(self, message: str, to_file: bool = True):
        """Log loss/negative message (red)."""
        self._log(f"[v] {message}", style="loss", to_file=to_file)

    def warning(self, message: str, to_file: bool = True):
        """Log warning message (yellow)."""
        self._log(f"[!] {message}", style="warning", to_file=to_file)

    def info(self, message: str, to_file: bool = True):
        """Log info message (cyan)."""
        self._log(f"[i] {message}", style="info", to_file=to_file)

    def header(self, message: str, to_file: bool = True):
        """Log header message (magenta)."""
        self._log(message, style="header", to_file=to_file)

    def metric(self, message: str, to_file: bool = True):
        """Log metric/statistic message (blue)."""
        self._log(message, style="metric", to_file=to_file)

    def neutral(self, message: str, to_file: bool = True):
        """Log neutral message (white)."""
        self._log(message, style="neutral", to_file=to_file)

    def dim(self, message: str, to_file: bool = True):
        """Log dimmed message (dim white)."""
        self._log(message, style="dim", to_file=to_file)

    def separator(self, char: str = "=", length: int = 80, to_file: bool = True):
        """Print a separator line."""
        self._log(char * length, to_file=to_file)

    def blank(self, to_file: bool = True):
        """Print a blank line."""
        self._log("", to_file=to_file)


# Global logger instance for convenience
_global_logger = Logger()


def get_logger(log_file: Optional[Path] = None) -> Logger:
    """
    Get a logger instance.

    Args:
        log_file: Optional file path to write logs to disk

    Returns:
        Logger instance
    """
    if log_file:
        return Logger(log_file=log_file)
    return _global_logger


# Convenience functions for global logger
def success(message: str, to_file: bool = False):
    """Log success message using global logger."""
    _global_logger.success(message, to_file=to_file)


def profit(message: str, to_file: bool = False):
    """Log profit message using global logger."""
    _global_logger.profit(message, to_file=to_file)


def error(message: str, to_file: bool = False):
    """Log error message using global logger."""
    _global_logger.error(message, to_file=to_file)


def loss(message: str, to_file: bool = False):
    """Log loss message using global logger."""
    _global_logger.loss(message, to_file=to_file)


def warning(message: str, to_file: bool = False):
    """Log warning message using global logger."""
    _global_logger.warning(message, to_file=to_file)


def info(message: str, to_file: bool = False):
    """Log info message using global logger."""
    _global_logger.info(message, to_file=to_file)


def header(message: str, to_file: bool = False):
    """Log header message using global logger."""
    _global_logger.header(message, to_file=to_file)


def metric(message: str, to_file: bool = False):
    """Log metric message using global logger."""
    _global_logger.metric(message, to_file=to_file)


def neutral(message: str, to_file: bool = False):
    """Log neutral message using global logger."""
    _global_logger.neutral(message, to_file=to_file)


def dim(message: str, to_file: bool = False):
    """Log dim message using global logger."""
    _global_logger.dim(message, to_file=to_file)


def separator(char: str = "=", length: int = 80, to_file: bool = False):
    """Print separator line using global logger."""
    _global_logger.separator(char, length, to_file=to_file)


def blank(to_file: bool = False):
    """Print blank line using global logger."""
    _global_logger.blank(to_file=to_file)
