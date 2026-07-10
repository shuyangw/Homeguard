"""#36 equity-index inter-market RV (NQ/ES, RTY/ES).

Pre-registered as 12-1 relative momentum on the log ratio (NOT mean reversion:
factor momentum is the evidenced direction). Beta-balance is deferred to the
book-correlation reporting: the make-or-break check is correlation to the
existing S&P equity-momentum sleeve, reported alongside the gate verdict. A high
positive correlation = re-expression / no marginal value even if Sharpe clears.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.spreads.construction import SpreadLeg, build_spread, round_trip_cost_usd
from src.backtesting.spreads.continuous import momentum_forecast, continuous_return_stream
from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.backtesting.walkforward_common import gate_return_stream
from src.utils.logger import get_logger

logger = get_logger(__name__)

PAIRS: dict[str, tuple[str, str]] = {
    "NQ_ES": ("NQ", "ES"),
    "RTY_ES": ("RTY", "ES"),
}


def intermarket_spread(long_root: str, short_root: str, start: date, end: date):
    panel = load_daily_panel([long_root, short_root], start, end)
    closes = panel.xs("close", axis=1, level=1)
    legs = [SpreadLeg(long_root, 1.0), SpreadLeg(short_root, -1.0)]
    return build_spread(legs, closes, mode="multiplicative")


def intermarket_return_stream(pair: str, start: date, end: date) -> pd.Series:
    long_root, short_root = PAIRS[pair]
    spread = intermarket_spread(long_root, short_root, start, end)
    forecast = momentum_forecast(spread.signal, lookback=252, skip=21, cap=2.0)
    cost = round_trip_cost_usd([SpreadLeg(long_root, 1.0), SpreadLeg(short_root, -1.0)])
    return continuous_return_stream(spread, forecast, cost_usd=cost)


def run_intermarket(pair: str, start: date, end: date, output_dir,
                    book_returns: pd.Series | None = None) -> dict:
    returns = intermarket_return_stream(pair, start, end)
    # gate_return_stream's walk-forward window math adds pd.DateOffset to the
    # index, which requires a DatetimeIndex; the panel loader's index is
    # python datetime.date objects.
    returns.index = pd.to_datetime(returns.index)
    result = gate_return_stream(returns)
    if book_returns is not None:
        aligned = pd.concat([returns, book_returns], axis=1, join="inner").dropna()
        result["book_corr"] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 2 else float("nan")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out / "returns.csv", header=True)
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    logger.info(f"[intermarket:{pair}] {result}")
    return result
