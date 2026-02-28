"""Portfolio snapshot logger for live trading.

Writes periodic portfolio snapshots to CSV for tracking equity curve,
position history, and daily P&L over time. Thread-safe for use from
background workers or multiple strategy threads.
"""

import csv
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger()

CSV_COLUMNS = [
    "timestamp",
    "equity",
    "cash",
    "buying_power",
    "num_positions",
    "total_unrealized_pnl",
    "positions",
]


class PortfolioLogger:
    """Logs portfolio snapshots to CSV and provides query methods.

    Each snapshot captures account equity, cash, buying power, and all
    open positions. Snapshots are appended to a single CSV file with
    thread-safe writes.

    Args:
        log_dir: Base directory for portfolio logs. A 'snapshots/'
                 subdirectory is created automatically.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._snapshots_dir = self._log_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._snapshots_dir / "portfolio_history.csv"
        self._lock = threading.Lock()
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        """Create CSV file with header row if it does not exist."""
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    @property
    def csv_path(self) -> Path:
        """Path to the portfolio history CSV file."""
        return self._csv_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_snapshot(
        self, account: Dict, positions: List[Dict]
    ) -> None:
        """Append a portfolio snapshot row to the CSV file.

        Args:
            account: Dict with keys 'equity', 'cash', 'buying_power'.
            positions: List of position dicts, each with 'symbol',
                       'quantity', 'current_price', 'unrealized_pnl',
                       'market_value'.

        This method never raises. All errors are caught and logged.
        """
        try:
            timestamp = tz.iso_timestamp()
            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            buying_power = float(account.get("buying_power", 0))
            num_positions = len(positions)
            total_unrealized_pnl = sum(
                float(p.get("unrealized_pnl", 0)) for p in positions
            )
            positions_str = self._serialize_positions(positions)

            row = [
                timestamp,
                equity,
                cash,
                buying_power,
                num_positions,
                total_unrealized_pnl,
                positions_str,
            ]

            with self._lock:
                with open(
                    self._csv_path, "a", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

        except Exception as exc:
            logger.error(f"PortfolioLogger.log_snapshot failed: {exc}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_latest_snapshot(self) -> Optional[Dict]:
        """Return the most recent snapshot as a dict, or None if empty."""
        try:
            df = self._read_csv()
            if df.empty:
                return None
            last = df.iloc[-1]
            return last.to_dict()
        except Exception as exc:
            logger.error(f"PortfolioLogger.get_latest_snapshot failed: {exc}")
            return None

    def get_snapshot_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return snapshot history as a DataFrame, optionally filtered by date.

        Args:
            start_date: Include snapshots on or after this date (inclusive).
            end_date: Include snapshots on or before this date (inclusive).

        Returns:
            DataFrame with CSV_COLUMNS. Empty DataFrame if no data.
        """
        try:
            df = self._read_csv()
            if df.empty:
                return df

            if start_date is not None or end_date is not None:
                df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
                if start_date is not None:
                    df = df[df["_date"] >= start_date]
                if end_date is not None:
                    df = df[df["_date"] <= end_date]
                df = df.drop(columns=["_date"])

            return df.reset_index(drop=True)
        except Exception as exc:
            logger.error(
                f"PortfolioLogger.get_snapshot_history failed: {exc}"
            )
            return pd.DataFrame(columns=CSV_COLUMNS)

    def get_daily_summary(
        self, target_date: Optional[date] = None
    ) -> Dict:
        """Return a summary dict for a single trading day.

        Args:
            target_date: Date to summarise. Defaults to today (ET).

        Returns:
            Dict with keys: equity_first, equity_last, equity_high,
            equity_low, snapshot_count, date.
        """
        if target_date is None:
            target_date = tz.today()

        try:
            df = self.get_snapshot_history(
                start_date=target_date, end_date=target_date
            )
            if df.empty:
                return {
                    "date": target_date.isoformat(),
                    "equity_first": None,
                    "equity_last": None,
                    "equity_high": None,
                    "equity_low": None,
                    "snapshot_count": 0,
                }

            equities = pd.to_numeric(df["equity"], errors="coerce")
            return {
                "date": target_date.isoformat(),
                "equity_first": float(equities.iloc[0]),
                "equity_last": float(equities.iloc[-1]),
                "equity_high": float(equities.max()),
                "equity_low": float(equities.min()),
                "snapshot_count": len(df),
            }
        except Exception as exc:
            logger.error(f"PortfolioLogger.get_daily_summary failed: {exc}")
            return {
                "date": target_date.isoformat(),
                "equity_first": None,
                "equity_last": None,
                "equity_high": None,
                "equity_low": None,
                "snapshot_count": 0,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_positions(positions: List[Dict]) -> str:
        """Serialize positions to semicolon-separated SYMBOL:QTY@PRICE string."""
        if not positions:
            return ""
        parts = []
        for p in positions:
            symbol = p.get("symbol", "???")
            qty = p.get("quantity", 0)
            price = p.get("current_price", 0)
            parts.append(f"{symbol}:{qty}@{price}")
        return ";".join(parts)

    def _read_csv(self) -> pd.DataFrame:
        """Read the portfolio history CSV into a DataFrame."""
        if not self._csv_path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)

        df = pd.read_csv(self._csv_path)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)
        return df
