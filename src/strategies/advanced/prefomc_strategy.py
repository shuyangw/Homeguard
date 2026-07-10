"""#39 Pre-FOMC drift: long ES+NQ over the 24h into the 14:00 ET FOMC statement
(entry 14:00 ET the prior trading day -> exit 14:00 ET on the FOMC day)
(Lucca-Moench 2015). Sign long, pre-registered. Ma-Zhang (2020) find the drift
largely disappeared after 2015; run_prefomc reports pre/post-2015 subperiod
Sharpe as the decay test -- NOT a sign flip."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.backtesting.session.session_bars import load_session_bars
from src.backtesting.session.session_simulator import SessionTrade, simulate_session_returns
from src.backtesting.session.session_walkforward import aggregate_returns, gate_session_stream
from src.data.derivations.futures.macro_calendar import load_macro_calendar


def prefomc_trades(bars_by_root, fomc_dates) -> list[SessionTrade]:
    trades: list[SessionTrade] = []
    fomc = set(fomc_dates)
    for root, bars in bars_by_root.items():
        dates = sorted(bars.index)
        for i, d in enumerate(dates):
            if d in fomc and i > 0:
                trades.append(SessionTrade(root, dates[i - 1], "et_1400", d, "et_1400", 1.0))
    return trades


def _sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 5 or np.nanstd(x, ddof=1) == 0:
        return float("nan")
    return float(np.nanmean(x) / np.nanstd(x, ddof=1) * np.sqrt(252))


def run_prefomc(roots=("ES", "NQ")) -> dict:
    bars_by_root = {r: load_session_bars(r) for r in roots}
    fomc_dates = load_macro_calendar("fomc")
    trades = prefomc_trades(bars_by_root, fomc_dates)
    per_root = {r: simulate_session_returns([t for t in trades if t.root == r],
                                            bars_by_root, cost_mult=1.0) for r in roots}
    ret = aggregate_returns(per_root)
    gate = gate_session_stream(ret)
    pre = ret[ret.index.map(lambda d: d.year < 2015)]
    post = ret[ret.index.map(lambda d: d.year >= 2015)]
    decay = {"pre_2015_sharpe": _sharpe(pre.to_numpy()), "post_2015_sharpe": _sharpe(post.to_numpy()),
             "pre_n": int(pre.size), "post_n": int(post.size)}
    out = Path("output") / "backtests" / "session" / "prefomc"
    out.mkdir(parents=True, exist_ok=True)
    ret.to_frame("return").to_csv(out / "returns.csv", index_label="date")
    (out / "gate.json").write_text(json.dumps({"gate": gate, "decay": decay}, default=float, indent=2))
    return {"gate": gate, "decay": decay, "n_trades": len(trades)}
