"""Stateful target-weight backtest engine for RAMP Phase 4 research.

Single entry: run_variant(cfg, variant_spec) -> list[DailyRecord].
Tasks 7-11 will extend the loop with MTM, trades, costs, and regime tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Protocol

import pandas as pd

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.data import load_universe_panel


@dataclass
class HarnessState:
    """Mutable state carried across the day-by-day loop."""
    cash_usd: float
    positions: Dict[str, float] = field(default_factory=dict)
    realized_pnl_usd: float = 0.0
    turnover_to_date_usd: float = 0.0
    cost_to_date_usd: float = 0.0


@dataclass
class DailyRecord:
    """Immutable record of one trading day."""
    date: datetime
    regime: str
    target_weights: Dict[str, float]
    realized_weights: Dict[str, float]
    turnover_usd: float
    cost_usd: float
    portfolio_value: float
    daily_return: float


class VariantLike(Protocol):
    """Duck-typed variant spec; the real one is in variants.py."""
    id: str
    plan_fn: Callable[[datetime, HarnessState, pd.DataFrame, HarnessConfig], Dict[str, float]]


def _portfolio_value(state: HarnessState, prices: pd.Series) -> float:
    """Cash + sum(shares * price) over current positions."""
    total = state.cash_usd
    for sym, shares in state.positions.items():
        px = prices.get(sym)
        if px is None or pd.isna(px):
            continue
        total += shares * px
    return total


def run_variant(cfg: HarnessConfig, variant_spec: VariantLike) -> List[DailyRecord]:
    """Run one variant end-to-end through [cfg.start_date, cfg.end_date].

    Returns one DailyRecord per trading day.
    """
    panel = load_universe_panel(cfg.universe_csv, cfg.start_date, cfg.end_date)
    if panel.empty:
        raise RuntimeError(f'Empty panel for {cfg.start_date.date()}..{cfg.end_date.date()}')

    state = HarnessState(cash_usd=cfg.initial_capital)
    records: List[DailyRecord] = []
    prev_value: float = cfg.initial_capital

    for t, row in panel.iterrows():
        ts = t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
        if not (cfg.start_date <= ts <= cfg.end_date):
            continue
        target_weights = variant_spec.plan_fn(ts, state, panel, cfg)
        cur_value = _portfolio_value(state, row)
        daily_ret = (cur_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        records.append(DailyRecord(
            date=ts,
            regime='STUB',
            target_weights=dict(target_weights),
            realized_weights={},
            turnover_usd=0.0,
            cost_usd=0.0,
            portfolio_value=cur_value,
            daily_return=daily_ret,
        ))
        prev_value = cur_value

    return records
