"""Beta-weighted spread portfolio simulator.

Holds a book of 2-leg spreads as NET per-instrument positions, MTM daily,
rebalances to spread-vol-targeted beta-weighted targets on rebalance days, and
charges cost per instrument on the net diff (both legs of a spread are charged;
internally-offsetting positions net first). Reuses FxBacktestResult and mirrors
the spot simulator's MTM/leverage/bankruptcy pattern.
"""
from __future__ import annotations

import pandas as pd

from src.backtesting.engine.fx_spot_portfolio_simulator import FxBacktestResult
from src.backtesting.engine.spread_sizing import spread_leg_targets


def _lag_book(spread_book: dict, sigma_panel: dict, dates: list, lag: int):
    """Shift a signal book forward by `lag` trading days.

    A book entry keyed to date d was computed from data through the CLOSE of d,
    so executing it at d's close is a same-bar fill: the strategy trades on
    information it could not have acted on until the bar was over. Shifting the
    keys to d+lag makes the fill convention honest (signal at close of d, filled
    at close of d+lag). Entries whose shifted date falls past the end of the
    sample are DROPPED (the signal never became tradeable).
    """
    if lag <= 0:
        return spread_book, sigma_panel
    pos = {d: i for i, d in enumerate(dates)}
    shifted_book, shifted_sigma = {}, {}
    for src, dst in ((spread_book, shifted_book), (sigma_panel, shifted_sigma)):
        for d, payload in (src or {}).items():
            i = pos.get(d)
            if i is None:
                continue
            j = i + lag
            if j < len(dates):
                dst[dates[j]] = payload
    return shifted_book, shifted_sigma


class FxSpreadPortfolioSimulator:
    """Beta-weighted spread simulator.

    `execution_lag` defaults to 1 bar: a book built from data through the close
    of day i is filled at the close of day i+1. This is the honest convention
    and is deliberately the DEFAULT so a run is realistic unless someone opts
    out explicitly. Results produced before 2026-07-25 used lag=0 (same-bar
    fills) and are therefore optimistic -- see
    docs/progress/20260725_fx_kalman_hedge_ratio.md.
    """

    def __init__(self, initial_capital: float, cost_fn, rebalance: str = "weekly",
                 cost_mult: float = 1.0, leverage_cap: float = 4.0,
                 execution_lag: int = 1):
        self.capital = float(initial_capital)
        self.cost_fn = cost_fn
        self.rebalance = rebalance
        self.cost_mult = float(cost_mult)
        self.leverage_cap = float(leverage_cap)
        self.execution_lag = int(execution_lag)

    def _is_rebalance(self, d, prev_d) -> bool:
        if self.rebalance == "daily" or prev_d is None:
            return True
        if self.rebalance == "weekly":
            return d.isocalendar()[1] != prev_d.isocalendar()[1]
        if self.rebalance == "monthly":
            return d.month != prev_d.month
        return True

    def _scale_leverage(self, targets: dict, close_row, q_row, equity: float) -> dict:
        gross = sum(abs(u * close_row[p] * q_row[p]) for p, u in targets.items())
        cap = self.leverage_cap * equity
        if gross > cap and gross > 0:
            f = cap / gross
            return {p: u * f for p, u in targets.items()}
        return targets

    def run_spreads(self, close_panel, spread_book, sigma_panel, quote_usd_panel,
                    vol_target: float, idm: float = 1.0) -> FxBacktestResult:
        pairs = list(close_panel.columns)
        dates = list(close_panel.index)
        spread_book, sigma_panel = _lag_book(spread_book, sigma_panel, dates,
                                             self.execution_lag)
        # The action grid must shift with the book. Gating on the UNSHIFTED
        # rebalance schedule would silently drop every lagged signal (a book
        # entry moved to d+1 lands on a non-rebalance day and is never read),
        # turning execution_lag into "trade nothing" instead of "trade later".
        reb_positions = []
        _prev = None
        for i, d in enumerate(dates):
            if self._is_rebalance(d, _prev):
                reb_positions.append(i)
            _prev = d
        lag = self.execution_lag
        action_positions = {i + lag for i in reb_positions if i + lag < len(dates)}

        current: dict[str, float] = {p: 0.0 for p in pairs}
        equity_val = self.capital
        equity, util, trade_rows = [], [], []
        prev_close, prev_d, blown = None, None, False

        for idx_d, d in enumerate(dates):
            row_close = {p: float(close_panel.loc[d, p]) for p in pairs}
            row_q = {p: float(quote_usd_panel.loc[d, p]) for p in pairs}
            # 1. MTM: pnl from close-to-close on held units (USD).
            # A pair with a NaN current/prior close or quote is forward-held
            # (0 P&L that day), not allowed to poison the whole sum.
            if prev_close is not None and not blown:
                pnl = 0.0
                for p in pairs:
                    u = current[p]
                    if u == 0.0:
                        continue
                    px, ppx, q = row_close[p], prev_close[p], row_q[p]
                    if pd.notna(px) and pd.notna(ppx) and pd.notna(q):
                        pnl += u * (px - ppx) * q
                equity_val += pnl
            if not blown and equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True
            # 2. Rebalance to spread targets. Pairs with a NaN close or quote
            # this day are excluded from targets, so they are forward-held.
            if not blown and idx_d in action_positions:
                spreads = spread_book.get(d, [])
                sigma = sigma_panel.get(d, {})
                raw = spread_leg_targets(spreads, sigma, row_close, row_q,
                                         equity_val, vol_target, idm)
                targets = {p: raw.get(p, 0.0) for p in pairs
                           if pd.notna(row_close[p]) and pd.notna(row_q[p])}
                targets = self._scale_leverage(targets, row_close, row_q, equity_val)
                for p in targets:
                    diff = targets[p] - current[p]
                    if diff != 0.0:
                        c = self.cost_fn(p, diff, row_close[p], row_q[p]) * self.cost_mult
                        equity_val -= c
                        trade_rows.append({"date": d, "pair": p, "units": diff, "cost": c})
                        current[p] = targets[p]
            if not blown and equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True
            gross = sum(abs(current[p] * row_close[p] * row_q[p]) for p in pairs
                        if pd.notna(row_close[p]) and pd.notna(row_q[p]))
            util.append(gross / equity_val if equity_val > 0 else 0.0)
            equity.append(equity_val)
            prev_close, prev_d = row_close, d

        eq = pd.Series(equity, index=dates, name="equity")
        lu = pd.Series(util, index=dates, name="leverage_utilization")
        trades = pd.DataFrame(trade_rows, columns=["date", "pair", "units", "cost"])
        return FxBacktestResult(equity_curve=eq, trades=trades, leverage_utilization=lu)
