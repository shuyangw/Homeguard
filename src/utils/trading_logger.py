"""
Production logging setup for trading bot with log rotation.

This module provides a configured logger specifically for live trading
with automatic log rotation to prevent disk space issues.

Usage:
    from src.utils.trading_logger import get_trading_logger

    logger = get_trading_logger()
    logger.info("Trading bot started")
    logger.error("Failed to execute trade", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


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

    # Log file with date prefix for easy identification
    today = datetime.now().strftime("%Y%m%d")
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
    formatter = logging.Formatter(
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

    # Separate file for executions
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"executions_{today}.log"

    # Rotating file handler (5MB per file, 10 backups)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=10,  # 50MB total
        encoding='utf-8'
    )

    # Simple format for execution logs (easier to parse)
    formatter = logging.Formatter(
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

    cutoff_date = datetime.now() - timedelta(days=keep_days)
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
