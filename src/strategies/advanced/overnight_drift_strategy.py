"""#21/#25 Overnight drift: long ES+NQ from the 16:00 ET cash close to the next
09:30 ET open. Nearly all the long-run index premium accrues overnight (Lou-Polk-
Skouras; NY-Fed SR 917). Sign is long, pre-registered; return net of one
round-trip/day (the 1.5x cost gate is the adjudicator). The next trading day is
the next date present in the session-bars cache (skips weekends/holidays)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting.session.session_bars import load_session_bars
from src.backtesting.session.session_simulator import SessionTrade, simulate_session_returns
from src.backtesting.session.session_walkforward import aggregate_returns, gate_session_stream


def overnight_trades(bars_by_root) -> list[SessionTrade]:
    trades: list[SessionTrade] = []
    for root, bars in bars_by_root.items():
        dates = list(bars.index)
        for i in range(len(dates) - 1):
            trades.append(SessionTrade(root, dates[i], "et_1600", dates[i + 1], "et_0930", 1.0))
    return trades


def run_overnight_drift(roots=("ES", "NQ")) -> dict:
    bars_by_root = {r: load_session_bars(r) for r in roots}
    trades = overnight_trades(bars_by_root)
    per_root_1x, per_root_15 = {}, {}
    for r in roots:
        rt = [t for t in trades if t.root == r]
        per_root_1x[r] = simulate_session_returns(rt, bars_by_root, cost_mult=1.0)
        per_root_15[r] = simulate_session_returns(rt, bars_by_root, cost_mult=1.5)
    ret_1x = aggregate_returns(per_root_1x)
    ret_15 = aggregate_returns(per_root_15)
    # gate_session_stream's walk-forward window arithmetic needs a DatetimeIndex
    # (Timestamp + DateOffset); aggregate_returns yields a python-date object index.
    ret_1x.index = pd.to_datetime(ret_1x.index)
    ret_15.index = pd.to_datetime(ret_15.index)
    gate = gate_session_stream(ret_1x)
    gate_15 = gate_session_stream(ret_15)
    out = Path("output") / "backtests" / "session" / "overnight_drift"
    out.mkdir(parents=True, exist_ok=True)
    ret_1x.to_frame("return").to_csv(out / "returns.csv", index_label="date")
    (out / "gate.json").write_text(json.dumps(
        {"gate_1x": gate, "gate_1_5x": gate_15}, default=float, indent=2))
    return {"gate_1x": gate, "gate_1_5x": gate_15, "n_trades": len(trades)}
