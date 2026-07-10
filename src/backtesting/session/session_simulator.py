"""Session-trade simulator: net-of-cost per-trade RETURNS (no contract sim).

Each SessionTrade names an entry and exit session-boundary close (from the
session-bars cache); the return is sign*(exit-entry)/entry minus a cost return
(the futures round-trip cost expressed as a fraction of notional). A NaN close
(missing bar / holiday) skips the trade -- never fabricate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.backtesting.costs.futures import futures_round_trip_usd
from src.data.futures.contract_specs import SPECS


@dataclass
class SessionTrade:
    root: str
    entry_date: date
    entry_col: str
    exit_date: date
    exit_col: str
    sign: float


def _cost_return(root: str, entry_close: float, cost_mult: float) -> float:
    notional = entry_close * SPECS[root].multiplier
    return futures_round_trip_usd(root, n_contracts=1) * cost_mult / notional


def simulate_session_returns(
    trades: list[SessionTrade],
    bars_by_root: dict[str, pd.DataFrame],
    cost_mult: float = 1.0,
) -> pd.Series:
    out: dict[date, float] = {}
    for tr in trades:
        bars = bars_by_root.get(tr.root)
        if bars is None or tr.entry_date not in bars.index or tr.exit_date not in bars.index:
            continue
        entry = bars.at[tr.entry_date, tr.entry_col]
        exit_ = bars.at[tr.exit_date, tr.exit_col]
        if not np.isfinite(entry) or not np.isfinite(exit_) or entry == 0.0:
            continue
        raw = tr.sign * (exit_ - entry) / entry
        out[tr.exit_date] = raw - _cost_return(tr.root, float(entry), cost_mult)
    return pd.Series(out).sort_index()
