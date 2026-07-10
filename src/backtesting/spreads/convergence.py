"""Convergence-trade spread engine ([D] strategies #31-#34).

A z-score state machine: enter toward convergence at |z|>entry_z, exit on
convergence (z->0 or sign flip), a time stop, or a structural-break stop. The
structural stop is ASYMMETRIC: tighter on the short (short-the-spread) side,
where a soft convenience-yield floor lets stretches trend. Sign is fixed by
construction (fade the stretch); never flipped post-hoc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtesting.walkforward_common import gate_return_stream


@dataclass(frozen=True)
class SpreadTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int      # -1 short the spread, +1 long the spread
    entry_z: float
    exit_z: float
    ret: float          # net-of-cost cumulative return over the hold


def rolling_z(signal: pd.Series, window: int) -> pd.Series:
    mean = signal.rolling(window).mean().shift(1)
    std = signal.rolling(window).std().shift(1)
    return (signal - mean) / std.replace(0.0, np.nan)


def simulate_convergence(signal: pd.Series, unit_return: pd.Series,
                         cost_return: float, window: int, entry_z: float = 2.0,
                         structural_z: float = 4.0, structural_z_short: float = 3.0,
                         max_hold: int = 60, converge_z: float = 0.25):
    z = rolling_z(signal, window)
    idx = signal.index
    daily = pd.Series(0.0, index=idx)
    trades: list[SpreadTrade] = []

    in_trade = False
    direction = 0
    entry_i = 0
    entry_zval = 0.0
    cum = 0.0

    for i in range(len(idx)):
        zi = z.iloc[i]
        if not in_trade:
            if np.isfinite(zi) and abs(zi) >= entry_z:
                candidate_direction = -int(np.sign(zi))   # fade the stretch
                candidate_bound = structural_z_short if candidate_direction < 0 else structural_z
                # Don't open a position that is already past its own structural
                # stop -- an entry only counts if the stretch is still inside
                # the band it will be exited from.
                if abs(zi) < candidate_bound:
                    in_trade = True
                    direction = candidate_direction
                    entry_i = i
                    entry_zval = zi
                    cum = 0.0
            continue

        # accrue P&L for the held position (position taken at entry, causal)
        step = direction * float(unit_return.iloc[i]) if np.isfinite(unit_return.iloc[i]) else 0.0
        cum += step
        daily.iloc[i] = step

        hold = i - entry_i
        bound = structural_z_short if direction < 0 else structural_z
        converged = (not np.isfinite(zi)) or abs(zi) < converge_z or np.sign(zi) != np.sign(entry_zval)
        broke = np.isfinite(zi) and abs(zi) > bound
        timed = hold >= max_hold

        if converged or broke or timed:
            net = cum - cost_return
            daily.iloc[i] = daily.iloc[i] - cost_return
            trades.append(SpreadTrade(idx[entry_i], idx[i], direction,
                                      float(entry_zval), float(zi) if np.isfinite(zi) else float("nan"),
                                      float(net)))
            in_trade = False
            direction = 0

    if in_trade:
        # Force-close a position still open at the end of the data so its
        # P&L isn't silently discarded (no explicit exit signal fired).
        net = cum - cost_return
        daily.iloc[-1] -= cost_return
        last_z = z.iloc[-1]
        trades.append(SpreadTrade(idx[entry_i], idx[-1], direction,
                                  float(entry_zval), float(last_z) if np.isfinite(last_z) else float("nan"),
                                  float(net)))

    return trades, daily.rename("return")


def gate_convergence(daily_returns: pd.Series) -> dict:
    return gate_return_stream(daily_returns)
