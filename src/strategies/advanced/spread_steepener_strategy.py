"""#35 yield-curve steepener/flattener via Micro Yield futures.

Signal = slope = long_tenor_yield - short_tenor_yield (close IS yield for these
roots). forecast = -z(slope) over a policy-cycle window (positive = steepener).
DV01-neutral: 1 long-tenor contract + 1 short-tenor contract ($10/bp each).

SIGN: pre-registered from theory and verified against a known steepening episode
(2023-11..2024-09 bull steepening as the Fed pivoted). A long-steepener position
(forecast > 0 when the curve is unusually FLAT/inverted, i.e. slope below its
mean) must earn POSITIVE cumulative return across that episode. If the empirical
check contradicts the registered sign, STOP and report -- do not flip the sign.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.spreads.construction import SpreadLeg, build_spread, round_trip_cost_usd
from src.backtesting.spreads.continuous import zscore_mr_forecast, continuous_return_stream
from src.backtesting.data.futures_backtest_loader import load_daily_panel
from src.backtesting.walkforward_common import gate_return_stream
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEGMENTS: dict[str, tuple[str, str]] = {
    "2s10s": ("2YY", "10Y"),
    "2s5s": ("2YY", "5YY"),
    "5s30s": ("5YY", "30Y"),
}


_FFILL_LIMIT_DAYS = 5  # bridge short non-trading/no-print gaps only (<=1 trading week)


def steepener_spread(short_root: str, long_root: str, start: date, end: date):
    panel = load_daily_panel([short_root, long_root], start, end)
    closes = panel.xs("close", axis=1, level=1)
    # Micro Yield futures have scattered single-day no-print gaps (thin/new
    # contracts); the underlying yield is still continuously quoted elsewhere,
    # so a short-limit forward-fill is standard data hygiene. This is capped at
    # 5 business days and applied uniformly to every leg BEFORE any signal is
    # built, so it cannot bridge 5YY's multi-month degraded stretches (see
    # docstring) -- those remain correctly NaN and ungradeable.
    closes = closes.ffill(limit=_FFILL_LIMIT_DAYS)
    legs = [SpreadLeg(long_root, 1.0), SpreadLeg(short_root, -1.0)]
    return build_spread(legs, closes, mode="additive")


def steepener_return_stream(segment: str, start: date, end: date,
                            window_days: int = 756) -> pd.Series:
    short_root, long_root = SEGMENTS[segment]
    spread = steepener_spread(short_root, long_root, start, end)
    forecast = zscore_mr_forecast(spread.signal, window=window_days, cap=2.0)
    cost = round_trip_cost_usd([SpreadLeg(long_root, 1.0), SpreadLeg(short_root, -1.0)])
    return continuous_return_stream(spread, forecast, cost_usd=cost)


def run_steepener(segment: str, start: date, end: date, output_dir) -> dict:
    returns = steepener_return_stream(segment, start, end)
    # gate_return_stream's walk-forward window math adds pd.DateOffset to the
    # index, which requires a DatetimeIndex; the panel loader's index is
    # python datetime.date objects.
    returns.index = pd.to_datetime(returns.index)
    result = gate_return_stream(returns)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    returns.to_csv(out / "returns.csv", header=True)
    (out / "gate.json").write_text(json.dumps(result, indent=2))
    logger.info(f"[steepener:{segment}] {result}")
    return result
