"""
Trade logging with configurable verbosity.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from visualization.config import LogLevel
from utils import logger as main_logger


class TradeEvent:
    """Represents a single trade event."""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        action: str,  # 'BUY' or 'SELL'
        price: float,
        size: float,
        portfolio_value: float,
        cash: float,
        position_value: float
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.action = action
        self.price = price
        self.size = size
        self.portfolio_value = portfolio_value
        self.cash = cash
        self.position_value = position_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'action': self.action,
            'price': self.price,
            'size': self.size,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': self.position_value
        }


class TradeLogger:
    """
    Logs trading activity with configurable verbosity.
    """

    def __init__(self, log_level: LogLevel = LogLevel.NORMAL):
        """
        Initialize trade logger.

        Args:
            log_level: Logging verbosity level
        """
        self.log_level = log_level
        self.trade_events: List[TradeEvent] = []
        self.log_buffer: List[str] = []
        self.console_trade_count = 0  # Track trades printed to console
        self.max_console_trades = 3  # Show first 3 and last 3 only
        self.suppression_message_shown = False
        self.last_trade_printed = False  # Track if last trade was printed to console

    def log_trade(self, event: TradeEvent):
        """
        Log a trade event.

        Args:
            event: TradeEvent instance
        """
        self.trade_events.append(event)

        if self.log_level.value >= LogLevel.NORMAL.value:
            self._log_normal(event)

        if self.log_level.value >= LogLevel.DETAILED.value:
            self._log_detailed(event)

    def _log_normal(self, event: TradeEvent):
        """Log at NORMAL level (trade entries/exits)."""
        timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp_str}] {event.action} {event.symbol}: {event.size:.2f} shares @ ${event.price:.2f}"
        self.log_buffer.append(msg)

        # Only print first 3 trades to console
        if self.console_trade_count < self.max_console_trades:
            # Color-code based on action
            if event.action == 'BUY':
                main_logger.profit(msg, to_file=False)
            else:
                main_logger.loss(msg, to_file=False)
            self.console_trade_count += 1
            self.last_trade_printed = True

            # Show suppression message after 3rd trade
            if self.console_trade_count == self.max_console_trades and not self.suppression_message_shown:
                self.suppression_message_shown = True
                main_logger.dim("\n... (additional trades hidden - see log file for complete details) ...\n", to_file=False)
        else:
            self.last_trade_printed = False

    def _log_detailed(self, event: TradeEvent):
        """Log at DETAILED level (portfolio changes)."""
        msg = (
            f"  Portfolio Value: ${event.portfolio_value:,.2f} | "
            f"Cash: ${event.cash:,.2f} | "
            f"Position Value: ${event.position_value:,.2f}"
        )
        self.log_buffer.append(msg)

        # Only print to console if the corresponding trade was printed
        if self.last_trade_printed:
            main_logger.dim(msg, to_file=False)

    def log_message(self, message: str, min_level: LogLevel = LogLevel.MINIMAL):
        """
        Log a general message.

        Args:
            message: Message to log
            min_level: Minimum log level required to display this message
        """
        if self.log_level.value >= min_level.value:
            self.log_buffer.append(message)
            main_logger.info(message, to_file=False)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get all trades as a DataFrame.

        Returns:
            DataFrame with trade information
        """
        if not self.trade_events:
            return pd.DataFrame()

        try:
            trades_data = [event.to_dict() for event in self.trade_events]

            # Extra defensive: ensure trades_data is a list of dicts
            if not isinstance(trades_data, list):
                return pd.DataFrame()

            result = pd.DataFrame(trades_data)

            # Defensive check - ensure we're returning a DataFrame, not a list
            if not isinstance(result, pd.DataFrame):
                return pd.DataFrame()

            return result
        except Exception as e:
            # If anything goes wrong, return empty DataFrame
            # Log the error for debugging
            main_logger.warning(f"Error creating trades DataFrame: {e}", to_file=False)
            return pd.DataFrame()

    def save_log(self, filepath: Path):
        """
        Save log to file and display last few trades to console.

        Args:
            filepath: Path to save log file
        """
        # Print last 3 trades to console if we suppressed any
        if len(self.trade_events) > self.max_console_trades:
            main_logger.blank(to_file=False)
            main_logger.dim("Last 3 trades:", to_file=False)
            for event in self.trade_events[-self.max_console_trades:]:
                timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                msg = f"[{timestamp_str}] {event.action} {event.symbol}: {event.size:.2f} shares @ ${event.price:.2f}"
                if event.action == 'BUY':
                    main_logger.profit(msg, to_file=False)
                else:
                    main_logger.loss(msg, to_file=False)

                # Show detailed info if at detailed level
                if self.log_level.value >= LogLevel.DETAILED.value:
                    detail_msg = (
                        f"  Portfolio Value: ${event.portfolio_value:,.2f} | "
                        f"Cash: ${event.cash:,.2f} | "
                        f"Position Value: ${event.position_value:,.2f}"
                    )
                    main_logger.dim(detail_msg, to_file=False)
            main_logger.blank(to_file=False)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Backtest Trade Log\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for line in self.log_buffer:
                f.write(line + "\n")

        main_logger.success(f"Log saved to: {filepath}")

    def save_trades_csv(self, filepath: Path):
        """
        Save trades to CSV file.

        Args:
            filepath: Path to save CSV file
        """
        df = self.get_trades_dataframe()

        # Defensive check: ensure df is actually a DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            main_logger.warning("No trades to save.")
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        main_logger.success(f"Trades CSV saved to: {filepath}")

    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of trades.

        Returns:
            Dictionary with trade summary statistics
        """
        df = self.get_trades_dataframe()

        # Defensive check: ensure df is actually a DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                'total_trades': 0,
                'buy_count': 0,
                'sell_count': 0,
            }

        return {
            'total_trades': len(df),
            'buy_count': len(df[df['action'] == 'BUY']),
            'sell_count': len(df[df['action'] == 'SELL']),
            'symbols_traded': df['symbol'].unique().tolist(),
            'total_volume': df['size'].sum(),
            'avg_trade_size': df['size'].mean(),
        }
