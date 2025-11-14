"""
Centralized logging module using Rich for colored console output.

All logging that is intended to be written to disk or displayed to users
must go through this module.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.theme import Theme
from datetime import datetime
import sys
import csv

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


class CSVLogger:
    """CSV file logger for structured data logging."""

    def __init__(self, filepath: Path, headers: List[str]):
        """
        Initialize CSV logger with headers.

        Args:
            filepath: Path to CSV file
            headers: Column headers for the CSV file
        """
        self.filepath = Path(filepath)
        self.headers = headers
        self._init_file()

    def _init_file(self):
        """Create CSV file with headers."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_row(self, data: List[Any]):
        """
        Append row to CSV file.

        Args:
            data: List of values matching headers length
        """
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data)


class TradingLogger:
    """
    Extended logger with file and CSV logging capabilities for trading.

    Combines colored console output, file logging, and structured CSV logs.
    Maintains same logging theme and cadence as existing code.
    """

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize trading logger.

        Args:
            name: Logger name (e.g., 'trading.OMR')
            log_dir: Directory for log files (optional)
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.csv_loggers: Dict[str, CSVLogger] = {}

        # Create base logger for console + file
        log_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}_{name}.log"

        self.logger = Logger(log_file=log_file)

    def add_csv_logger(self, name: str, filepath: Path, headers: List[str]):
        """
        Add a CSV logger for structured data.

        Args:
            name: Name for this CSV logger (e.g., 'trades', 'market_checks')
            filepath: Path to CSV file
            headers: Column headers
        """
        self.csv_loggers[name] = CSVLogger(filepath, headers)

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        status: str,
        order_type: str = 'market',
        order_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log trade to CSV and console with consistent formatting.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            qty: Quantity
            price: Price
            status: 'SUCCESS' or 'FAILED'
            order_type: Order type (default: 'market')
            order_id: Order ID (optional)
            error: Error message if failed (optional)
        """
        timestamp = datetime.now().isoformat()

        # Log to CSV if available
        if 'trades' in self.csv_loggers:
            self.csv_loggers['trades'].log_row([
                timestamp,
                symbol,
                side,
                qty,
                price,
                order_type,
                status,
                order_id if order_id else '',
                error if error else ''
            ])

        # Log to console with color
        if status == 'SUCCESS':
            self.logger.success(
                f"Trade: {symbol} {side.upper()} {qty} @ ${price:.2f}"
            )
        else:
            self.logger.error(
                f"Trade FAILED: {symbol} {side.upper()} {qty} @ ${price:.2f} - {error}"
            )

    def log_market_check(self, market_open: bool, check_number: int):
        """
        Log market check to CSV.

        Args:
            market_open: Whether market is open
            check_number: Check sequence number
        """
        timestamp = datetime.now().isoformat()

        # Log to CSV if available
        if 'market_checks' in self.csv_loggers:
            self.csv_loggers['market_checks'].log_row([
                timestamp,
                market_open,
                check_number
            ])

    # Expose base logger methods for convenience
    def success(self, message: str, to_file: bool = True):
        """Log success message (green)."""
        self.logger.success(message, to_file=to_file)

    def error(self, message: str, to_file: bool = True):
        """Log error message (red)."""
        self.logger.error(message, to_file=to_file)

    def warning(self, message: str, to_file: bool = True):
        """Log warning message (yellow)."""
        self.logger.warning(message, to_file=to_file)

    def info(self, message: str, to_file: bool = True):
        """Log info message (cyan)."""
        self.logger.info(message, to_file=to_file)

    def header(self, message: str, to_file: bool = True):
        """Log header message (magenta)."""
        self.logger.header(message, to_file=to_file)

    def separator(self, char: str = "=", length: int = 80, to_file: bool = True):
        """Print a separator line."""
        self.logger.separator(char, length, to_file=to_file)

    def blank(self, to_file: bool = True):
        """Print a blank line."""
        self.logger.blank(to_file=to_file)


# Global logger instance for convenience
_global_logger = Logger()

# Export logger for direct import
logger = _global_logger


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


def get_trading_logger(name: str, log_dir: Optional[Path] = None) -> TradingLogger:
    """
    Get a trading logger instance with CSV logging capabilities.

    Args:
        name: Logger name (e.g., 'trading.OMR', 'MACrossover')
        log_dir: Directory for log files (optional)

    Returns:
        TradingLogger instance

    Usage:
        # Create trading logger
        trading_logger = get_trading_logger('OMR', Path('logs/live_trading/paper'))

        # Add CSV loggers
        trading_logger.add_csv_logger(
            'trades',
            Path('logs/trades.csv'),
            ['timestamp', 'symbol', 'side', 'qty', 'price', 'status']
        )

        # Log trades
        trading_logger.log_trade('TQQQ', 'buy', 100, 45.23, 'SUCCESS')

        # Use regular logging methods
        trading_logger.success("Trade executed successfully")
        trading_logger.info("Market check complete")
    """
    return TradingLogger(name, log_dir)


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
