"""
Minute-level stop-loss simulation for overnight/intraday strategies.

Used for strategies like OMR that hold overnight and need realistic
stop-loss simulation during the next trading session.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader

logger = get_logger(__name__)


class IntradayStopSimulator:
    """
    Simulates stop-loss triggers using minute-level data.

    For strategies that enter positions near market close and exit
    the next morning, this simulator checks minute bars to determine
    if stop-losses would have been triggered.

    Usage:
        simulator = IntradayStopSimulator(stop_loss_pct=0.02)
        updated_trades = simulator.simulate_stops(trades_df)
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        trailing_stop_pct: Optional[float] = None,
        data_loader: Optional[StreamingDataLoader] = None,
    ):
        """
        Initialize simulator.

        Args:
            stop_loss_pct: Fixed stop loss as decimal (e.g., 0.02 = 2%)
            trailing_stop_pct: Trailing stop as decimal (optional)
            data_loader: Data loader for minute bars (creates new if None)
        """
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.data_loader = data_loader or StreamingDataLoader()
        self._minute_cache: Dict[str, pd.DataFrame] = {}

    def simulate_stops(
        self,
        trades_df: pd.DataFrame,
        entry_time: str = "15:50",
        exit_time: str = "09:31",
        check_overnight: bool = True,
        direction: str = "long",
    ) -> pd.DataFrame:
        """
        Check each trade for stop-loss triggers using minute data.

        Args:
            trades_df: DataFrame with columns:
                - entry_timestamp or timestamp
                - symbol
                - entry_price or price
            entry_time: Time of entry (for reference)
            exit_time: Default exit time if no stop triggered
            check_overnight: If True, check pre-market next day
            direction: "long" or "short"

        Returns:
            DataFrame with updated columns:
            - exit_timestamp: Actual exit time (stop or scheduled)
            - exit_price: Price at exit
            - exit_reason: 'stop_loss', 'trailing_stop', 'scheduled_exit'
            - stop_triggered: Boolean
            - exit_bar: Minute bar index where exit occurred
            - pnl: Recalculated P&L if stop triggered
            - pnl_pct: Recalculated P&L %
        """
        if trades_df.empty:
            return trades_df

        # Normalize column names
        df = trades_df.copy()
        if "entry_timestamp" not in df.columns and "timestamp" in df.columns:
            df["entry_timestamp"] = df["timestamp"]
        if "entry_price" not in df.columns and "price" in df.columns:
            df["entry_price"] = df["price"]

        results = []
        for idx, trade in df.iterrows():
            result = self._check_single_trade(
                trade, entry_time, exit_time, check_overnight, direction
            )
            results.append(result)

        result_df = pd.DataFrame(results)

        # Merge with original to preserve any extra columns
        for col in ["exit_timestamp", "exit_price", "exit_reason", "stop_triggered", "exit_bar"]:
            if col in result_df.columns:
                df[col] = result_df[col].values

        # Recalculate P&L for stop-triggered trades
        if "stop_triggered" in df.columns:
            mask = df["stop_triggered"] == True
            if mask.any():
                if direction == "long":
                    df.loc[mask, "pnl"] = (
                        df.loc[mask, "exit_price"] - df.loc[mask, "entry_price"]
                    ) * df.loc[mask, "shares"]
                    df.loc[mask, "pnl_pct"] = (
                        (df.loc[mask, "exit_price"] - df.loc[mask, "entry_price"])
                        / df.loc[mask, "entry_price"]
                        * 100
                    )
                else:  # short
                    df.loc[mask, "pnl"] = (
                        df.loc[mask, "entry_price"] - df.loc[mask, "exit_price"]
                    ) * df.loc[mask, "shares"]
                    df.loc[mask, "pnl_pct"] = (
                        (df.loc[mask, "entry_price"] - df.loc[mask, "exit_price"])
                        / df.loc[mask, "entry_price"]
                        * 100
                    )

        return df

    def _check_single_trade(
        self,
        trade: pd.Series,
        entry_time: str,
        exit_time: str,
        check_overnight: bool,
        direction: str,
    ) -> Dict[str, Any]:
        """Check single trade for stop triggers."""
        symbol = trade.get("symbol", "UNKNOWN")
        entry_ts = pd.to_datetime(trade.get("entry_timestamp", trade.get("timestamp")))
        entry_price = float(trade.get("entry_price", trade.get("price", 0)))
        entry_date = entry_ts.date()

        # Get next trading day
        next_day = self._get_next_trading_day(entry_date)

        # Load minute data for the exit day
        minute_data = self._load_minute_data(symbol, next_day)

        if minute_data is None or minute_data.empty:
            # No minute data available - return scheduled exit
            return {
                "exit_timestamp": pd.Timestamp(f"{next_day} {exit_time}"),
                "exit_price": entry_price,
                "exit_reason": "scheduled_exit",
                "stop_triggered": False,
                "exit_bar": -1,
            }

        # Filter to market hours (up to exit time)
        exit_hour, exit_minute = map(int, exit_time.split(":"))
        exit_cutoff = pd.Timestamp(f"{next_day} {exit_time}")
        minute_data = minute_data[minute_data.index <= exit_cutoff]

        if minute_data.empty:
            return {
                "exit_timestamp": exit_cutoff,
                "exit_price": entry_price,
                "exit_reason": "scheduled_exit",
                "stop_triggered": False,
                "exit_bar": -1,
            }

        # Check each minute bar for stop trigger
        stop_price = entry_price * (1 - self.stop_loss_pct) if direction == "long" else entry_price * (1 + self.stop_loss_pct)
        highest_price = entry_price
        lowest_price = entry_price

        for bar_idx, (ts, row) in enumerate(minute_data.iterrows()):
            # For longs: check if low <= stop price
            # For shorts: check if high >= stop price
            if direction == "long":
                check_price = row["low"] if "low" in row else row["close"]
                if check_price <= stop_price:
                    return {
                        "exit_timestamp": ts,
                        "exit_price": stop_price,
                        "exit_reason": "stop_loss",
                        "stop_triggered": True,
                        "exit_bar": bar_idx,
                    }

                # Trailing stop (if enabled)
                if self.trailing_stop_pct is not None:
                    high = row["high"] if "high" in row else row["close"]
                    highest_price = max(highest_price, high)
                    trailing_stop = highest_price * (1 - self.trailing_stop_pct)
                    if check_price <= trailing_stop:
                        return {
                            "exit_timestamp": ts,
                            "exit_price": trailing_stop,
                            "exit_reason": "trailing_stop",
                            "stop_triggered": True,
                            "exit_bar": bar_idx,
                        }
            else:  # short
                check_price = row["high"] if "high" in row else row["close"]
                if check_price >= stop_price:
                    return {
                        "exit_timestamp": ts,
                        "exit_price": stop_price,
                        "exit_reason": "stop_loss",
                        "stop_triggered": True,
                        "exit_bar": bar_idx,
                    }

                # Trailing stop for shorts
                if self.trailing_stop_pct is not None:
                    low = row["low"] if "low" in row else row["close"]
                    lowest_price = min(lowest_price, low)
                    trailing_stop = lowest_price * (1 + self.trailing_stop_pct)
                    if check_price >= trailing_stop:
                        return {
                            "exit_timestamp": ts,
                            "exit_price": trailing_stop,
                            "exit_reason": "trailing_stop",
                            "stop_triggered": True,
                            "exit_bar": bar_idx,
                        }

        # No stop triggered - exit at scheduled time
        exit_ts = minute_data.index[-1] if len(minute_data) > 0 else exit_cutoff
        exit_price = minute_data.iloc[-1]["close"] if len(minute_data) > 0 else entry_price

        return {
            "exit_timestamp": exit_ts,
            "exit_price": exit_price,
            "exit_reason": "scheduled_exit",
            "stop_triggered": False,
            "exit_bar": len(minute_data) - 1,
        }

    def _get_next_trading_day(self, date) -> datetime:
        """Get next trading day after given date."""
        from pandas.tseries.offsets import BDay

        next_day = pd.Timestamp(date) + BDay(1)
        return next_day.date()

    def _load_minute_data(self, symbol: str, date) -> Optional[pd.DataFrame]:
        """Load minute data for symbol on given date."""
        cache_key = f"{symbol}_{date}"

        if cache_key in self._minute_cache:
            return self._minute_cache[cache_key]

        try:
            df = self.data_loader.load_symbol(
                symbol,
                start_date=str(date),
                end_date=str(date),
                timeframe="1min",
            )

            if df is not None and not df.empty:
                self._minute_cache[cache_key] = df
                return df

        except Exception as e:
            logger.debug(f"Could not load minute data for {symbol} on {date}: {e}")

        return None

    def clear_cache(self):
        """Clear the minute data cache."""
        self._minute_cache.clear()


def simulate_intraday_stops(
    trades_df: pd.DataFrame,
    stop_loss_pct: float = 0.02,
    trailing_stop_pct: Optional[float] = None,
    entry_time: str = "15:50",
    exit_time: str = "09:31",
    direction: str = "long",
) -> pd.DataFrame:
    """
    Convenience function to simulate intraday stops.

    Args:
        trades_df: DataFrame with entry_timestamp, symbol, entry_price
        stop_loss_pct: Fixed stop loss percentage
        trailing_stop_pct: Optional trailing stop percentage
        entry_time: Entry time for reference
        exit_time: Default exit time
        direction: "long" or "short"

    Returns:
        DataFrame with updated exit information
    """
    simulator = IntradayStopSimulator(
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )

    return simulator.simulate_stops(
        trades_df,
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
    )
