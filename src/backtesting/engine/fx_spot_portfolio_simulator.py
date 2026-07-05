"""Daily spot-FX backtest simulator with carry accrual.

Separate from the equity and futures simulators. Positions are signed notionals
in units of each pair's base currency. Each day: mark-to-market in USD, accrue
the interest-rate differential (carry) on held USD notional, debit costs on
rebalance days, then enforce a gross-notional leverage cap and a bankruptcy
floor (equity is provably non-negative).

    base_to_usd  = price * quote_to_usd            # 1 base unit in USD
    mtm_usd      = sum_p units_p * (px_t - px_{t-1}) * quote_to_usd_t
    carry_usd    = sum_p (units_p * base_to_usd_t) * rate_diff_t / 365
    equity_t     = equity_{t-1} + mtm_usd + carry_usd - costs_t
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtesting.utils.position_sizer_fx import size_from_forecast_fx

CostFn = Callable[[str, float, float, float], float]  # (pair, units_traded, price, quote_to_usd)


@dataclass
class FxBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    leverage_utilization: pd.Series


class FxSpotPortfolioSimulator:
    def __init__(self, initial_capital: float, cost_fn: CostFn,
                 rebalance: str = "daily", cost_mult: float = 1.0,
                 leverage_cap: float = 10.0):
        self.initial_capital = float(initial_capital)
        self.cost_fn = cost_fn
        self.rebalance = rebalance
        self.cost_mult = float(cost_mult)
        self.leverage_cap = float(leverage_cap)

    def _is_rebalance(self, d, prev_d) -> bool:
        if self.rebalance == "daily" or prev_d is None:
            return True
        if self.rebalance == "weekly":
            return d.isocalendar()[1] != prev_d.isocalendar()[1]
        if self.rebalance == "monthly":
            return d.month != prev_d.month
        return True

    def _scale_to_leverage(self, targets: dict, base_to_usd: dict, equity: float) -> dict:
        gross = sum(abs(u * base_to_usd[p]) for p, u in targets.items())
        cap_notional = self.leverage_cap * equity
        if gross <= cap_notional or gross <= 0:
            return targets
        scale = cap_notional / gross
        return {p: u * scale for p, u in targets.items()}

    def run_sized(self, close_panel: pd.DataFrame, forecast_panel: pd.DataFrame,
                  daily_vol_panel: pd.DataFrame, vol_target: float,
                  quote_usd_panel: pd.DataFrame, rate_diff_panel: pd.DataFrame,
                  div_mult: float | dict = 1.0) -> FxBacktestResult:
        pairs = list(close_panel.columns)
        dates = list(close_panel.index)
        equity_val = self.initial_capital
        current = {p: 0.0 for p in pairs}
        equity, util, trade_rows = [], [], []
        prev_close = None
        prev_d = None
        blown = False

        def dm(p):
            return div_mult if isinstance(div_mult, (int, float)) else div_mult.get(p, 1.0)

        for d in dates:
            row_close = close_panel.loc[d]
            row_q = quote_usd_panel.loc[d]
            row_rd = rate_diff_panel.loc[d]

            if blown:
                util.append(0.0)
                equity.append(0.0)
                prev_close, prev_d = row_close, d
                continue

            # 1. MTM + carry on existing positions
            if prev_close is not None:
                pnl = 0.0
                for p in pairs:
                    u = current[p]
                    if u == 0.0:
                        continue
                    px, ppx, q = row_close[p], prev_close[p], row_q[p]
                    if pd.notna(px) and pd.notna(ppx) and pd.notna(q):
                        pnl += u * (px - ppx) * q
                    if pd.notna(px) and pd.notna(q) and pd.notna(row_rd[p]):
                        pnl += (u * px * q) * row_rd[p] / 365.0
                equity_val += pnl

            if equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True
                util.append(0.0)
                equity.append(0.0)
                prev_close, prev_d = row_close, d
                continue

            # 2. Rebalance -> target notionals, leverage-capped
            if self._is_rebalance(d, prev_d):
                base_to_usd = {}
                targets = {}
                for p in pairs:
                    px, q = row_close[p], row_q[p]
                    f = forecast_panel.loc[d, p] if d in forecast_panel.index else float("nan")
                    v = daily_vol_panel.loc[d, p] if d in daily_vol_panel.index else float("nan")
                    if pd.isna(px) or pd.isna(q) or pd.isna(f) or pd.isna(v):
                        base_to_usd[p] = 0.0
                        targets[p] = 0.0
                        continue
                    b2u = px * q
                    base_to_usd[p] = b2u
                    targets[p] = size_from_forecast_fx(
                        float(f), equity_val, vol_target, b2u, float(v), dm(p))
                targets = self._scale_to_leverage(targets, base_to_usd, equity_val)
                for p in pairs:
                    want = targets[p]
                    diff = want - current[p]
                    if diff != 0.0:
                        c = self.cost_fn(p, diff, float(row_close[p]), float(row_q[p])) * self.cost_mult
                        equity_val -= c
                        trade_rows.append({"date": d, "pair": p, "units": diff, "cost": c})
                        current[p] = want

            if not blown and equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True

            gross = sum(abs(current[p] * (row_close[p] * row_q[p]))
                        for p in pairs if pd.notna(row_close[p]) and pd.notna(row_q[p]))
            util.append(gross / equity_val if equity_val > 0 else 0.0)
            equity.append(equity_val)
            prev_close, prev_d = row_close, d

        eq = pd.Series(equity, index=dates, name="equity")
        lu = pd.Series(util, index=dates, name="leverage_utilization")
        trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["date", "pair", "units", "cost"])
        return FxBacktestResult(equity_curve=eq, trades=trades, leverage_utilization=lu)
