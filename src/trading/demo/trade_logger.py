"""
Trade logging for demo trading.

Logs all trades in JSON Lines format with portfolio snapshots,
enabling detailed post-hoc analysis of trading activity.
"""

import json
import threading
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

from src.trading.demo.execution_simulator import ExecutionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _json_encoder(obj):
    """JSON encoder for special types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class DemoTradeLogger:
    """
    Logs demo trades to JSON Lines files.

    Each trade is logged with:
    - Trade details (symbol, side, quantity, prices)
    - Execution details (slippage, fees, latency)
    - Portfolio snapshot at time of trade

    Files are rotated daily for easy analysis.

    File location: ~/.homeguard/demo/trades/YYYY-MM-DD.jsonl

    Log format (per line):
    {
        "timestamp": "2025-01-07T10:30:00-05:00",
        "trade_id": "uuid",
        "symbol": "BTC/USD",
        "side": "buy",
        "quantity": "0.001",
        "requested_price": 95000.0,
        "fill_price": 95004.75,
        "slippage_bps": 5.0,
        "fees": 9.50,
        "latency_ms": 87,
        "portfolio_snapshot": {
            "total_value": 10500.0,
            "cash": 5000.0,
            "positions": {"BTC/USD": 0.001},
            "unrealized_pnl": 150.0
        }
    }
    """

    DEFAULT_LOG_DIR = Path.home() / ".homeguard" / "demo" / "trades"

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        include_snapshots: bool = True,
    ):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory for trade logs (default: ~/.homeguard/demo/trades/)
            include_snapshots: Whether to include portfolio snapshots
        """
        self._log_dir = log_dir or self.DEFAULT_LOG_DIR
        self._include_snapshots = include_snapshots
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._current_date: Optional[str] = None
        self._trade_count = 0

        self._ensure_log_dir()
        logger.info(f"[TradeLogger] Logging trades to {self._log_dir}")

    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """Get current log file (rotates daily)."""
        today = datetime.now().strftime("%Y-%m-%d")

        if self._current_date != today:
            self._current_date = today
            self._current_file = self._log_dir / f"{today}.jsonl"
            logger.info(f"[TradeLogger] Using log file: {self._current_file}")

        return self._current_file

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        execution: ExecutionResult,
        portfolio_snapshot: Optional[Dict] = None,
        order_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log a trade to the trade log.

        Args:
            symbol: Crypto symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            execution: Execution result from simulator
            portfolio_snapshot: Portfolio state after trade
            order_id: Optional order ID
            metadata: Optional additional metadata

        Returns:
            Trade ID (UUID)
        """
        with self._lock:
            trade_id = str(uuid.uuid4())

            # Calculate slippage in basis points
            if execution.market_price > 0:
                slippage_bps = (
                    abs(execution.fill_price - execution.market_price)
                    / execution.market_price
                    * 10000
                )
            else:
                slippage_bps = 0.0

            entry = {
                "timestamp": execution.timestamp.isoformat(),
                "trade_id": trade_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "requested_price": execution.market_price,
                "fill_price": execution.fill_price,
                "slippage_bps": round(slippage_bps, 2),
                "slippage_cost": round(execution.slippage_cost, 4),
                "fees": round(execution.fees, 4),
                "total_cost": round(execution.total_cost, 4),
                "latency_ms": execution.latency_ms,
            }

            if self._include_snapshots and portfolio_snapshot:
                entry["portfolio_snapshot"] = portfolio_snapshot

            if metadata:
                entry["metadata"] = metadata

            # Write to file
            try:
                log_file = self._get_log_file()
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry, default=_json_encoder) + "\n")

                self._trade_count += 1

                logger.debug(
                    f"[TradeLogger] Logged trade {trade_id[:8]}: {side.upper()} {quantity} {symbol}"
                )

            except Exception as e:
                logger.error(f"[TradeLogger] Error logging trade: {e}")

            return trade_id

    def get_trades(
        self,
        date: Optional[str] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list:
        """
        Read trades from log files.

        Args:
            date: Date in YYYY-MM-DD format (None = today)
            symbol: Filter by symbol
            side: Filter by side ('buy' or 'sell')
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        log_file = self._log_dir / f"{date}.jsonl"

        if not log_file.exists():
            return []

        trades = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    trade = json.loads(line)

                    # Apply filters
                    if symbol and trade.get("symbol") != symbol:
                        continue
                    if side and trade.get("side") != side:
                        continue

                    trades.append(trade)

                    if limit and len(trades) >= limit:
                        break

        except Exception as e:
            logger.error(f"[TradeLogger] Error reading trades: {e}")

        return trades

    def get_trade_summary(self, date: Optional[str] = None) -> Dict:
        """
        Get summary statistics for a day's trades.

        Args:
            date: Date in YYYY-MM-DD format (None = today)

        Returns:
            Dict with summary statistics
        """
        trades = self.get_trades(date=date)

        if not trades:
            return {
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "total_trades": 0,
                "buys": 0,
                "sells": 0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "symbols_traded": [],
            }

        buys = [t for t in trades if t["side"] == "buy"]
        sells = [t for t in trades if t["side"] == "sell"]
        symbols = list(set(t["symbol"] for t in trades))

        total_fees = sum(t.get("fees", 0) for t in trades)
        total_slippage = sum(t.get("slippage_cost", 0) for t in trades)

        return {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "total_fees": round(total_fees, 2),
            "total_slippage": round(total_slippage, 2),
            "symbols_traded": symbols,
            "avg_latency_ms": round(
                sum(t.get("latency_ms", 0) for t in trades) / len(trades), 1
            ),
        }

    def get_available_dates(self) -> list:
        """Get list of dates with trade logs."""
        dates = []
        for f in self._log_dir.glob("*.jsonl"):
            date_str = f.stem
            dates.append(date_str)
        return sorted(dates)

    @property
    def trade_count(self) -> int:
        """Get total trades logged in this session."""
        return self._trade_count

    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return self._log_dir
