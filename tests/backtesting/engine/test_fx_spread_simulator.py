import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.engine.spread_sizing import Spread
from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator


def _flat_cost(pair, units, price, q):
    return abs(units) * price * q * 0.0001  # 1bp of notional per trade, simple


def _panel(pairs, n=40):
    idx = pd.date_range("2022-01-03", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    data = {p: 1.0 + np.cumsum(rng.normal(0, 0.002, n)) for p in pairs}
    return pd.DataFrame(data, index=pd.Index(idx))


def test_both_legs_charged_on_entry():
    pairs = ["AUDUSD", "NZDUSD"]
    close = _panel(pairs)
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    d0 = close.index[0]
    book = {d: [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)] for d in close.index}
    sigma = {d: {("AUDUSD", "NZDUSD"): 0.01} for d in close.index}
    sim = FxSpreadPortfolioSimulator(100000.0, _flat_cost, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    first_rebal = res.trades[res.trades["date"] == res.trades["date"].min()]
    assert set(first_rebal["pair"]) == {"AUDUSD", "NZDUSD"}  # BOTH legs traded/charged


def test_market_neutral_no_pnl_when_legs_move_together():
    # If both legs move identically and beta=1, the spread (A-B) has ~0 PnL.
    pairs = ["AUDUSD", "NZDUSD"]
    idx = pd.date_range("2022-01-03", periods=30, freq="B").date
    common = 1.0 + np.cumsum(np.full(30, 0.001))  # identical path
    close = pd.DataFrame({"AUDUSD": common, "NZDUSD": common}, index=pd.Index(idx))
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    book = {d: [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)] for d in idx}
    sigma = {d: {("AUDUSD", "NZDUSD"): 0.01} for d in idx}
    sim = FxSpreadPortfolioSimulator(100000.0, lambda *a: 0.0, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    # equity barely moves (legs cancel): final within 0.5% of start
    assert abs(res.equity_curve.iloc[-1] / 100000.0 - 1.0) < 0.005


def test_empty_book_holds_flat():
    pairs = ["AUDUSD", "NZDUSD"]
    close = _panel(pairs)
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    book = {d: [] for d in close.index}
    sigma = {d: {} for d in close.index}
    sim = FxSpreadPortfolioSimulator(100000.0, _flat_cost, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    assert (res.equity_curve == 100000.0).all()  # never traded
