"""
CSCM signal logger for hypothetical performance tracking.

Logs hourly signal snapshots and weekly rebalance decisions to JSONL files.
These records allow reconstruction of hypothetical portfolio performance
without requiring a live broker connection.

Log files: {log_dir}/cscm/cscm_signals_YYYYMMDD.jsonl
"""

import json
from pathlib import Path
from typing import Dict, List

from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)


class CSCMSignalLogger:
    """
    Logs CSCM strategy signals to daily-rotated JSONL files.

    Records two types of entries:
    - signal: Hourly snapshot of regime, BTC price/SMA, momentum rankings
    - rebalance: Weekly target portfolio with position weights
    """

    def __init__(self, log_dir: Path):
        self._log_dir = Path(log_dir) / "cscm"
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"[CSCMSignalLogger] Failed to create log dir: {e}")

        logger.info(f"[CSCMSignalLogger] Signals -> {self._log_dir}")

    def _get_log_file(self) -> Path:
        """Get daily-rotated log file path."""
        date_str = tz.now().strftime("%Y%m%d")
        return self._log_dir / f"cscm_signals_{date_str}.jsonl"

    def _write_record(self, record: Dict) -> None:
        """Append a JSON record to the current day's log file."""
        try:
            log_file = self._get_log_file()
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"[CSCMSignalLogger] Failed to write record: {e}")

    def log_signal(
        self,
        regime: str,
        btc_price: float,
        btc_sma: float,
        is_rebalance_day: bool,
        reduce_exposure: bool,
        exposure_pct: float,
        top_symbols: List[str],
        momentum_scores: Dict[str, float],
    ) -> None:
        """Log hourly signal snapshot."""
        record = {
            "timestamp": tz.iso_timestamp(),
            "type": "signal",
            "regime": regime,
            "btc_price": btc_price,
            "btc_sma": btc_sma,
            "is_rebalance_day": is_rebalance_day,
            "reduce_exposure": reduce_exposure,
            "exposure_pct": exposure_pct,
            "top_symbols": top_symbols,
            "momentum_scores": momentum_scores,
        }
        self._write_record(record)

    def log_rebalance(
        self,
        regime: str,
        btc_price: float,
        btc_sma: float,
        top_symbols: List[str],
        momentum_scores: Dict[str, float],
        target_positions: Dict[str, float],
    ) -> None:
        """Log rebalance decision with target weights."""
        record = {
            "timestamp": tz.iso_timestamp(),
            "type": "rebalance",
            "regime": regime,
            "btc_price": btc_price,
            "btc_sma": btc_sma,
            "top_symbols": top_symbols,
            "momentum_scores": momentum_scores,
            "target_positions": target_positions,
        }
        self._write_record(record)
