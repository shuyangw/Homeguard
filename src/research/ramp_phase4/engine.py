"""Stateful target-weight backtest engine for RAMP Phase 4 research.

Single entry: run_variant(cfg, variant_spec) -> list[DailyRecord].
Tasks 7-11 will extend the loop with MTM, trades, costs, and regime tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Protocol

import pandas as pd

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.costs import flat_bps_cost
from src.research.ramp_phase4.data import load_universe_panel


OVERLEVERAGE_EPSILON = 1e-3


@dataclass
class HarnessState:
    """Mutable state carried across the day-by-day loop."""
    cash_usd: float
    positions: Dict[str, float] = field(default_factory=dict)
    realized_pnl_usd: float = 0.0
    turnover_to_date_usd: float = 0.0
    cost_to_date_usd: float = 0.0
    # Phase C Wave 1 additions:
    position_open_dates: Dict[str, datetime] = field(default_factory=dict)
    last_target_symbols: List[str] = field(default_factory=list)


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
    """Run one variant end-to-end through [cfg.start_date, cfg.end_date]."""
    panel = load_universe_panel(cfg.universe_csv, cfg.start_date, cfg.end_date)
    if panel.empty:
        raise RuntimeError(f'Empty panel for {cfg.start_date.date()}..{cfg.end_date.date()}')

    state = HarnessState(cash_usd=cfg.initial_capital)
    records: List[DailyRecord] = []
    prev_value: float = cfg.initial_capital

    for t, prices in panel.iterrows():
        ts = t.to_pydatetime()
        if not (cfg.start_date <= ts <= cfg.end_date):
            continue

        # 1. MTM + forced exits.
        _handle_nan_exits(state, prices)
        cur_value = _portfolio_value(state, prices)

        # 2. Variant produces target weights + optional regime sentinel.
        plan_output = variant_spec.plan_fn(ts, state, panel, cfg) or {}
        regime_label = str(plan_output.pop('__regime__', 'STUB'))
        target_weights = plan_output

        if regime_label == 'SAFE_MODE':
            # Hold current positions; no trades.
            post_value = cur_value
            daily_ret = (post_value / prev_value) - 1.0 if prev_value > 0 else 0.0
            records.append(DailyRecord(
                date=ts, regime=regime_label,
                target_weights=dict(target_weights),
                realized_weights=_current_weights(state, prices, post_value),
                turnover_usd=0.0, cost_usd=0.0,
                portfolio_value=post_value, daily_return=daily_ret,
            ))
            prev_value = post_value
            continue

        weight_sum = sum(target_weights.values())
        if weight_sum > 1.0 + OVERLEVERAGE_EPSILON:
            raise ValueError(
                f'Variant {variant_spec.id} returned overleverage at {ts.date()}: '
                f'sum(weights)={weight_sum:.4f}'
            )

        # 3. Compute trades + cost, apply.
        trades = compute_trades(state, target_weights, prices, cur_value, cfg.min_trade_value_usd)
        cost = flat_bps_cost(trades, cfg.cost_bps_per_side)
        turnover = sum(abs(t['trade_value_usd']) for t in trades)
        apply_trades(state, trades, cost_usd=cost, current_date=ts)

        # 4. Post-trade MTM.
        post_value = _portfolio_value(state, prices)
        daily_ret = (post_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        realized_weights = _current_weights(state, prices, post_value)

        records.append(DailyRecord(
            date=ts, regime=regime_label,
            target_weights=dict(target_weights),
            realized_weights=realized_weights,
            turnover_usd=turnover,
            cost_usd=cost,
            portfolio_value=post_value,
            daily_return=daily_ret,
        ))
        prev_value = post_value

    return records


def _handle_nan_exits(state: HarnessState, prices: pd.Series) -> List[str]:
    """If a held symbol has NaN price today, exit at last known good price (already in cash).

    Returns the list of forced-exit symbols.
    """
    exits: List[str] = []
    for sym in list(state.positions.keys()):
        px = prices.get(sym)
        if px is None or pd.isna(px):
            # Position becomes worthless in MTM; treat as forced exit at 0 cash impact.
            # Caller (engine) can choose richer behavior later; for Phase B this matches
            # the "force-exit at last good close" intent -- by the time we see NaN, we've
            # already recorded yesterday's MTM at the last good price.
            del state.positions[sym]
            exits.append(sym)
    return exits


def _current_weights(state: HarnessState, prices: pd.Series, total_value: float) -> Dict[str, float]:
    """Map symbol -> current weight (position value / total portfolio value)."""
    if total_value <= 0:
        return {}
    out: Dict[str, float] = {}
    for sym, shares in state.positions.items():
        px = prices.get(sym)
        if px is None or pd.isna(px):
            continue
        out[sym] = (shares * px) / total_value
    return out


def compute_trades(
    state: HarnessState,
    target_weights: Dict[str, float],
    prices: pd.Series,
    total_value: float,
    min_trade_value_usd: float,
) -> List[Dict]:
    """Compute the delta trades from current positions to target weights.

    Whole-share rounding. Drops trades below `min_trade_value_usd`.
    Sells appear with negative trade_value_usd; buys positive.

    Each trade dict: {symbol, delta_shares, trade_value_usd, side}.
    """
    if total_value <= 0:
        return []

    # Build union of current + target symbols.
    all_syms = set(state.positions.keys()) | set(target_weights.keys())
    trades: List[Dict] = []
    for sym in all_syms:
        px = prices.get(sym)
        if px is None or pd.isna(px) or px <= 0:
            continue
        current_shares = state.positions.get(sym, 0.0)
        target_value = target_weights.get(sym, 0.0) * total_value
        target_shares_float = target_value / px
        target_shares = math.floor(target_shares_float) if target_shares_float >= 0 else math.ceil(target_shares_float)
        delta = target_shares - current_shares
        if delta == 0:
            continue
        trade_value = delta * px
        if abs(trade_value) < min_trade_value_usd:
            continue
        trades.append({
            'symbol': sym,
            'delta_shares': int(delta),
            'trade_value_usd': float(trade_value),
            'side': 'buy' if delta > 0 else 'sell',
        })
    return trades


def apply_trades(
    state: HarnessState,
    trades: List[Dict],
    cost_usd: float,
    current_date: datetime,
) -> None:
    """Mutate state to apply the given trades + cost.

    Sells execute first to free cash. Buys execute in input order (caller
    should sort by rank if cash is tight). After this call, state.cash_usd
    is decreased by total buy notional + cost, increased by total sell notional.

    Symbols with zero shares are removed from state.positions.

    position_open_dates bookkeeping:
    - Set when a position transitions from 0 shares to >0 shares (new open).
    - Preserved on top-ups (existing position grows; original open date stays).
    - Cleared when a position transitions to 0 shares (full exit).
    """
    sells = [t for t in trades if t['delta_shares'] < 0]
    buys = [t for t in trades if t['delta_shares'] > 0]
    for t in sells + buys:
        sym = t['symbol']
        current = state.positions.get(sym, 0.0)
        new_shares = current + t['delta_shares']
        state.cash_usd -= t['trade_value_usd']
        if new_shares == 0:
            state.positions.pop(sym, None)
            state.position_open_dates.pop(sym, None)
        else:
            state.positions[sym] = float(new_shares)
            if current == 0:
                state.position_open_dates[sym] = current_date
        state.turnover_to_date_usd += abs(t['trade_value_usd'])
    state.cash_usd -= cost_usd
    state.cost_to_date_usd += cost_usd
