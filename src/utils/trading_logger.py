"""
Production logging setup for trading bot with log rotation.

This module provides a configured logger specifically for live trading
with automatic log rotation to prevent disk space issues.

Usage:
    from src.utils.trading_logger import get_trading_logger, get_trade_log_writer

    logger = get_trading_logger()
    logger.info("Trading bot started")
    logger.error("Failed to execute trade", exc_info=True)

    # Structured trade logging
    trade_logger = get_trade_log_writer()
    trade_logger.log_entry(strategy='omr', symbol='TQQQ', qty=100, price=68.42, order_id='abc123')
    trade_logger.log_exit(strategy='omr', symbol='TQQQ', qty=100, exit_price=69.15, ...)
"""

import json
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, List, Any
import time

from src.utils.timezone import tz
from src.utils.logger import logger as system_logger


class ESTFormatter(logging.Formatter):
    """
    Custom logging formatter that uses EST/ET timezone for timestamps.

    This ensures consistent EST timestamps regardless of system timezone.
    """

    def formatTime(self, record, datefmt=None):
        """Override formatTime to use EST timezone."""
        # Convert the record's created time (UTC timestamp) to EST
        ct = tz.from_utc(datetime.utcfromtimestamp(record.created))
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime('%Y-%m-%d %H:%M:%S')


def get_trading_logger(
    name: str = "homeguard-trading",
    log_dir: str = "/home/ec2-user/logs",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5  # Keep 5 backup files (50MB total)
) -> logging.Logger:
    """
    Get a configured logger for the trading bot with automatic rotation.

    Features:
    - Logs to rotating file (10MB per file, 5 backups = 50MB max)
    - Also logs to console for real-time monitoring
    - Structured format with timestamps
    - ERROR and above always logged, INFO optional

    Args:
        name: Logger name (default: "homeguard-trading")
        log_dir: Directory for log files (default: /home/ec2-user/logs)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Max size per log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_trading_logger()
        >>> logger.info("Bot started")
        >>> logger.error("Order failed", exc_info=True)
    """
    # Create logger
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Log file with date prefix for easy identification (EST date)
    today = tz.date_str()
    log_file = log_path / f"trading_{today}.log"

    # ===== FILE HANDLER (Rotating) =====
    # Rotates after 10MB, keeps 5 backup files
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file

    # ===== CONSOLE HANDLER =====
    # Only show INFO and above in console (less noise during SSH)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # ===== FORMATTER =====
    # Format: [2025-01-15 15:50:23] [INFO] [homeguard-trading] Message here
    # Uses ESTFormatter to ensure consistent EST timestamps
    formatter = ESTFormatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ===== ADD HANDLERS =====
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # ===== LOG STARTUP INFO =====
    logger.info("=" * 80)
    logger.info(f"Trading Logger Initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Max size per file: {max_bytes / 1024 / 1024:.1f} MB")
    logger.info(f"Backup files: {backup_count}")
    logger.info(f"Max total log size: {max_bytes * (backup_count + 1) / 1024 / 1024:.1f} MB")
    logger.info("=" * 80)

    return logger


def get_execution_logger(log_dir: str = "/home/ec2-user/logs") -> logging.Logger:
    """
    Get a separate logger for trade execution events.

    This creates a dedicated log file for trades only, making it easy
    to audit execution history without application noise.

    Args:
        log_dir: Directory for log files

    Returns:
        Logger instance for trade execution

    Example:
        >>> exec_logger = get_execution_logger()
        >>> exec_logger.info("BUY TQQQ 100 shares @ $45.32")
    """
    logger = logging.getLogger("homeguard-execution")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Separate file for executions (EST date)
    today = tz.date_str()
    log_file = log_path / f"executions_{today}.log"

    # Rotating file handler (5MB per file, 10 backups)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,  # 50MB total
        encoding='utf-8'
    )

    # Simple format for execution logs (easier to parse)
    # Uses ESTFormatter to ensure consistent EST timestamps
    formatter = ESTFormatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def cleanup_old_logs(log_dir: str = "/home/ec2-user/logs", keep_days: int = 30):
    """
    Clean up log files older than specified days.

    Call this periodically (e.g., weekly) to remove old logs and save disk space.

    Args:
        log_dir: Directory containing log files
        keep_days: Number of days to keep (default: 30)

    Example:
        >>> cleanup_old_logs(keep_days=30)  # Remove logs older than 30 days
    """
    from datetime import timedelta

    log_path = Path(log_dir)
    if not log_path.exists():
        return

    cutoff_date = tz.now() - timedelta(days=keep_days)
    deleted_count = 0
    deleted_size = 0

    for log_file in log_path.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_date.timestamp():
            file_size = log_file.stat().st_size
            log_file.unlink()
            deleted_count += 1
            deleted_size += file_size

    if deleted_count > 0:
        logger = logging.getLogger("homeguard-trading")
        logger.info(f"Cleaned up {deleted_count} old log files ({deleted_size / 1024 / 1024:.2f} MB)")


# Convenience function for quick setup
def setup_trading_logs(log_dir: str = "/home/ec2-user/logs", log_level: str = "INFO"):
    """
    One-line setup for trading bot logging.

    Returns both main logger and execution logger.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (INFO, DEBUG, etc.)

    Returns:
        Tuple of (main_logger, execution_logger)

    Example:
        >>> logger, exec_logger = setup_trading_logs()
        >>> logger.info("Bot initialized")
        >>> exec_logger.info("BUY TQQQ 100 @ $45.32")
    """
    main_logger = get_trading_logger(log_dir=log_dir, log_level=log_level)
    exec_logger = get_execution_logger(log_dir=log_dir)

    return main_logger, exec_logger


# =============================================================================
# Trade Log Writer - Structured JSON Lines Trade Logging
# =============================================================================

class TradeLogWriter:
    """
    Persistent trade log writer using JSON Lines format.

    Writes completed trades to a daily log file with full details
    including entry/exit data and P&L for round-trip trades.

    Features:
    - JSON Lines format for easy parsing and append-only writes
    - Daily files: trades_{YYYYMMDD}.jsonl
    - Dual logging: writes to file AND system journal
    - Error-safe: logging failures never block trading

    Usage:
        trade_logger = get_trade_log_writer()
        trade_logger.log_entry(strategy='omr', symbol='TQQQ', qty=100, price=68.42, order_id='abc123')
        trade_logger.log_exit(strategy='omr', symbol='TQQQ', qty=100, exit_price=69.15, ...)
    """

    def __init__(self, log_dir: str = "/home/ec2-user/logs"):
        """
        Initialize trade log writer.

        Args:
            log_dir: Directory for trade log files (default: /home/ec2-user/logs)
        """
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            system_logger.error(f"[TRADE_LOG] Failed to create log directory {log_dir}: {e}")

    def _get_log_file(self) -> Path:
        """Get today's trade log file path."""
        today = tz.date_str()  # YYYYMMDD format
        return self.log_dir / f"trades_{today}.jsonl"

    def log_entry(
        self,
        strategy: str,
        symbol: str,
        qty: int,
        price: float,
        order_id: Optional[str] = None,
        order_type: str = "market",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trade entry (buy or short entry).

        Args:
            strategy: Strategy name ('omr' or 'mp')
            symbol: Stock symbol (e.g., 'TQQQ', 'AAPL')
            qty: Number of shares
            price: Entry price (filled price from broker)
            order_id: Broker order ID for traceability
            order_type: Order type ('market', 'limit', etc.)
            metadata: Optional additional data (rank, probability, etc.)
        """
        record = {
            "timestamp": tz.iso_timestamp(),
            "strategy": strategy,
            "symbol": symbol,
            "side": "buy",
            "qty": qty,
            "price": round(price, 4),
            "order_id": order_id,
            "order_type": order_type,
            "trade_type": "entry",
            "metadata": metadata or {}
        }
        self._write_record(record)

    def log_exit(
        self,
        strategy: str,
        symbol: str,
        qty: int,
        exit_price: float,
        order_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        entry_time: Optional[str] = None,
        order_type: str = "market",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trade exit with P&L calculation.

        Args:
            strategy: Strategy name ('omr' or 'mp')
            symbol: Stock symbol
            qty: Number of shares exited
            exit_price: Exit price (filled price from broker)
            order_id: Broker order ID
            entry_price: Original entry price for P&L calculation
            entry_time: Original entry timestamp
            order_type: Order type ('market', 'limit', etc.)
            metadata: Optional additional data
        """
        # Calculate P&L if entry price is available
        pnl_dollars = None
        pnl_pct = None
        if entry_price and entry_price > 0:
            pnl_dollars = round((exit_price - entry_price) * qty, 2)
            pnl_pct = round(((exit_price - entry_price) / entry_price) * 100, 4)

        record = {
            "timestamp": tz.iso_timestamp(),
            "strategy": strategy,
            "symbol": symbol,
            "side": "sell",
            "qty": qty,
            "price": round(exit_price, 4),
            "order_id": order_id,
            "order_type": order_type,
            "trade_type": "exit",
            "entry_price": round(entry_price, 4) if entry_price else None,
            "entry_time": entry_time,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "metadata": metadata or {}
        }
        self._write_record(record)

    def _write_record(self, record: Dict[str, Any]) -> None:
        """
        Append record to log file with error handling.

        CRITICAL: This method never raises exceptions - logging failures
        must not block trading execution.
        """
        try:
            log_file = self._get_log_file()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

            # System-level logging for real-time monitoring via journalctl
            trade_type = record.get('trade_type', 'unknown').upper()
            strategy = record.get('strategy', '?').upper()
            symbol = record.get('symbol', '?')
            qty = record.get('qty', 0)
            price = record.get('price', 0)

            if record.get('trade_type') == 'exit' and record.get('pnl_dollars') is not None:
                pnl = record.get('pnl_dollars', 0)
                pnl_pct = record.get('pnl_pct', 0)
                system_logger.info(
                    f"[TRADE_LOG] {trade_type}: {strategy} {symbol} {qty} @ ${price:.2f} "
                    f"(P&L: ${pnl:+.2f}, {pnl_pct:+.2f}%)"
                )
            else:
                system_logger.info(
                    f"[TRADE_LOG] {trade_type}: {strategy} {symbol} {qty} @ ${price:.2f}"
                )

        except Exception as e:
            # CRITICAL: Never fail silently - log errors but don't raise
            system_logger.error(f"[TRADE_LOG] Failed to write trade record: {e}")
            system_logger.error(f"[TRADE_LOG] Record data: {record}")
            # Don't raise - trade logging failure should not block trading


# Global singleton for easy access
_trade_log_writer: Optional[TradeLogWriter] = None


def get_trade_log_writer(log_dir: str = "/home/ec2-user/logs") -> TradeLogWriter:
    """
    Get singleton trade log writer instance.

    Args:
        log_dir: Directory for trade log files

    Returns:
        TradeLogWriter instance

    Example:
        >>> trade_logger = get_trade_log_writer()
        >>> trade_logger.log_entry(strategy='omr', symbol='TQQQ', qty=100, price=68.42)
    """
    global _trade_log_writer
    if _trade_log_writer is None:
        _trade_log_writer = TradeLogWriter(log_dir)
    return _trade_log_writer


def read_trade_log(date_str: Optional[str] = None, log_dir: str = "/home/ec2-user/logs") -> List[Dict]:
    """
    Read trade log for a specific date.

    Args:
        date_str: Date in YYYYMMDD format (defaults to today)
        log_dir: Log directory path

    Returns:
        List of trade records

    Example:
        >>> trades = read_trade_log()  # Today's trades
        >>> trades = read_trade_log('20251205')  # Specific date
        >>> for t in trades:
        ...     print(f"{t['symbol']}: {t['trade_type']} @ ${t['price']}")
    """
    if date_str is None:
        date_str = tz.date_str()

    log_file = Path(log_dir) / f"trades_{date_str}.jsonl"

    if not log_file.exists():
        return []

    trades = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
    except Exception as e:
        system_logger.error(f"[TRADE_LOG] Failed to read trade log {log_file}: {e}")

    return trades
