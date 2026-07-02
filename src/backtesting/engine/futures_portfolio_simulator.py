"""Daily multi-instrument futures backtest simulator.

Separate from the equity/crypto PortfolioSimulator. Per-contract daily
mark-to-market into cash; per-contract dollar costs on contracts traded
(position diff) only on rebalance days; margin utilization recorded per day.
Equity == cash (positions are MTM'd into cash each day).

Cost convention: `cost_fn(root, regular_hours, n_contracts)` returns the
TOTAL cost for `n_contracts` (matching `futures_round_trip_usd`, which
already scales by n_contracts). The simulator does NOT multiply by
n_contracts again.

Bankruptcy floor: if MTM drives cash <= 0, the broker force-liquidates all
positions, cash floors at 0.0, and the account stays flat (equity == 0.0)
for the rest of the series. This caps the account's loss at 100% and keeps
the equity curve well-defined (no negative equity, no divide-by-zero blowups
in downstream pct_change statistics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtesting.utils.position_sizer_futures import size_from_forecast
from src.data.futures.contract_specs import get_spec


def _floor(current: dict) -> tuple[float, bool]:
    """Force-liquidate all positions and floor cash at 0.0. Returns (cash, blown)."""
    for r in current:
        current[r] = 0
    return 0.0, True


@dataclass
class FuturesBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    margin_utilization: pd.Series


# target_provider(d, equity_now, current) -> dict[root, int] desired contracts
TargetProvider = Callable[[object, float, dict], dict]


class FuturesPortfolioSimulator:
    def __init__(self, initial_capital, cost_fn, margin_model,
                 rebalance: str = "weekly", cost_mult: float = 1.0):
        self.initial_capital = float(initial_capital)
        self.cost_fn = cost_fn
        self.margin = margin_model
        self.rebalance = rebalance
        self.cost_mult = float(cost_mult)

    def _is_rebalance(self, d, prev_d) -> bool:
        if self.rebalance == "daily":
            return True
        if prev_d is None:
            return True
        if self.rebalance == "weekly":
            return d.isocalendar().week != prev_d.isocalendar().week
        if self.rebalance == "monthly":
            return d.month != prev_d.month
        return True

    def _simulate(self, close_panel: pd.DataFrame, target_provider: TargetProvider) -> FuturesBacktestResult:
        roots = list(close_panel.columns)
        dates = list(close_panel.index)
        cash = self.initial_capital
        current = {r: 0 for r in roots}
        equity, util, trade_rows = [], [], []
        prev_close = None
        prev_d = None
        blown = False

        for d in dates:
            row_close = close_panel.loc[d]

            if blown:
                util.append(self.margin.utilization(current, cash))
                equity.append(cash)
                prev_close = row_close
                prev_d = d
                continue

            # 1. MTM on existing positions
            if prev_close is not None:
                pnl = 0.0
                for r in roots:
                    if current[r] != 0 and pd.notna(row_close[r]) and pd.notna(prev_close[r]):
                        pnl += current[r] * get_spec(r).multiplier * (row_close[r] - prev_close[r])
                cash += pnl

            # 2. Bankruptcy floor -- force-liquidate, flatten, floor cash at 0
            if cash <= 0:
                cash, blown = _floor(current)
                util.append(self.margin.utilization(current, cash))
                equity.append(cash)
                prev_close = row_close
                prev_d = d
                continue

            # 3. Rebalance
            if self._is_rebalance(d, prev_d):
                tgt = target_provider(d, cash, current)
                for r in roots:
                    val = tgt.get(r)
                    want = int(val) if val is not None and pd.notna(val) else 0
                    diff = want - current[r]
                    if diff != 0:
                        c = self.cost_fn(r, regular_hours=True, n_contracts=abs(diff)) * self.cost_mult
                        cash -= c
                        trade_rows.append({"date": d, "root": r, "contracts": diff, "cost": c})
                        current[r] = want

            # 3b. Bankruptcy floor -- catch cost-driven negative equity the same day
            if not blown and cash <= 0:
                cash, blown = _floor(current)

            # 4. Margin utilization
            util.append(self.margin.utilization(current, cash))
            equity.append(cash)
            prev_close = row_close
            prev_d = d

        eq = pd.Series(equity, index=dates, name="equity")
        um = pd.Series(util, index=dates, name="margin_utilization")
        trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["date", "root", "contracts", "cost"])
        return FuturesBacktestResult(equity_curve=eq, trades=trades, margin_utilization=um)

    def run(self, close_panel: pd.DataFrame, target_contracts: pd.DataFrame) -> FuturesBacktestResult:
        def provider(d, equity_now, current):
            return target_contracts.loc[d].to_dict()

        return self._simulate(close_panel, provider)

    def run_sized(self, close_panel: pd.DataFrame, forecast_panel: pd.DataFrame,
                  daily_vol_panel: pd.DataFrame, vol_target: float,
                  div_mult: float = 1.0) -> FuturesBacktestResult:
        roots = list(close_panel.columns)

        def provider(d, equity_now, current):
            row: dict[str, int] = {}
            for r in roots:
                forecast = forecast_panel.loc[d, r] if d in forecast_panel.index else float("nan")
                price = close_panel.loc[d, r]
                vol = daily_vol_panel.loc[d, r] if d in daily_vol_panel.index else float("nan")
                if pd.isna(forecast) or pd.isna(price) or pd.isna(vol):
                    row[r] = 0
                    continue
                row[r] = size_from_forecast(
                    float(forecast), equity_now, vol_target, r,
                    price=float(price), daily_vol=float(vol), div_mult=div_mult,
                )
            return self.margin.check_and_scale(row, equity=equity_now)

        return self._simulate(close_panel, provider)
