"""Daily multi-instrument futures backtest simulator.

Separate from the equity/crypto PortfolioSimulator. Per-contract daily
mark-to-market into cash; per-contract dollar costs on contracts traded
(position diff) only on rebalance days; margin utilization recorded per day.
Equity == cash (positions are MTM'd into cash each day).

Cost convention: `cost_fn(root, regular_hours, n_contracts)` returns the
TOTAL cost for `n_contracts` (matching `futures_round_trip_usd`, which
already scales by n_contracts). The simulator does NOT multiply by
n_contracts again.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.futures.contract_specs import get_spec


@dataclass
class FuturesBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    margin_utilization: pd.Series


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

    def run(self, close_panel: pd.DataFrame, target_contracts: pd.DataFrame) -> FuturesBacktestResult:
        roots = list(close_panel.columns)
        dates = list(close_panel.index)
        cash = self.initial_capital
        current = {r: 0 for r in roots}
        equity, util, trade_rows = [], [], []
        prev_close = None
        prev_d = None

        for d in dates:
            row_close = close_panel.loc[d]
            # 1. MTM on existing positions
            if prev_close is not None:
                pnl = 0.0
                for r in roots:
                    if current[r] != 0 and pd.notna(row_close[r]) and pd.notna(prev_close[r]):
                        pnl += current[r] * get_spec(r).multiplier * (row_close[r] - prev_close[r])
                cash += pnl

            # 2. Rebalance
            if self._is_rebalance(d, prev_d):
                tgt = target_contracts.loc[d]
                for r in roots:
                    want = int(tgt[r]) if pd.notna(tgt[r]) else 0
                    diff = want - current[r]
                    if diff != 0:
                        c = self.cost_fn(r, regular_hours=True, n_contracts=abs(diff)) * self.cost_mult
                        cash -= c
                        trade_rows.append({"date": d, "root": r, "contracts": diff, "cost": c})
                        current[r] = want

            # 3. Margin utilization
            util.append(self.margin.utilization(current, cash))
            equity.append(cash)
            prev_close = row_close
            prev_d = d

        eq = pd.Series(equity, index=dates, name="equity")
        um = pd.Series(util, index=dates, name="margin_utilization")
        trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["date", "root", "contracts", "cost"])
        return FuturesBacktestResult(equity_curve=eq, trades=trades, margin_utilization=um)
