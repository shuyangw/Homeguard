"""RAMP target planner -- Phase 4 F1.

Pure planning layer. Produces a `RampPlan` per rebalance describing the
INTENDED portfolio: target weights for every name to hold (entries + holds),
plus exits, plus regime diagnostics.

Consumed by:
- The live adapter (`src/trading/adapters/ramp_live_adapter.py`) when
  `use_target_planner=True`, to drive target-aware execution.
- The future stateful backtest engine (Phase 4 Phase B / F6).

Core sizing rule:
    target_weight = max_capital_allocation * exposure_pct / top_n

Existing positions not in the target set get target weight zero (force exit).
New BUY names and existing HOLD names both receive the same per-position
target weight.

This module is stateless: pure functions of inputs. No I/O, no logging beyond
raised exceptions for schema violations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RampTarget:
    """An intended position for a single symbol on a single rebalance day.

    Fields:
        symbol: ticker
        target_weight: 0.0 - max_capital_allocation, fraction of equity_base.
                       0.0 means "exit this position".
        rank: cross-sectional momentum rank at planning time (1 = best).
              None for exits.
        regime: the regime label used to derive the weight.
        reason: "new_entry" | "hold" | "topup" | "trim" | "exit".
    """
    symbol: str
    target_weight: float
    rank: Optional[int]
    regime: str
    reason: str


@dataclass(frozen=True)
class RampPlan:
    """A complete rebalance plan for a single trading day.

    Fields:
        as_of: planning timestamp.
        regime: label used (one of 5 RAMP regimes, or "SAFE_MODE").
        regime_confidence: detector confidence 0.0 - 1.0.
        regime_scores: dict of all five regime scores (not only the winner).
        exposure_pct: crash-protection multiplier applied (0.0, 0.5, or 1.0).
        top_n: target position count for this regime.
        targets: ALL intended positions (entries + holds), keyed by symbol.
                 Excludes exits (those go in `exits`).
        exits: symbol -> reason. Positions to fully exit this rebalance.
        diagnostics: VIX value, VIX percentile, SPY drawdown, breadth, etc.
                     Used by the decision log (F5).
    """
    as_of: datetime
    regime: str
    regime_confidence: float
    regime_scores: Dict[str, float]
    exposure_pct: float
    top_n: int
    targets: Dict[str, RampTarget]
    exits: Dict[str, str]
    diagnostics: Dict[str, Any]


def compute_plan(
    *,
    as_of: datetime,
    regime: str,
    regime_confidence: float,
    regime_scores: Dict[str, float],
    top_n: int,
    momentum_scores: "pd.Series",
    current_positions: Dict[str, float],
    vix: float,
    spy_drawdown: float,
    max_capital_allocation: float = 1.0,
    diagnostics: Optional[Dict[str, Any]] = None,
    vix_threshold: float = 25.0,
    spy_dd_threshold: float = -0.05,
    reduced_exposure: float = 0.5,
    safe_mode: bool = False,
    bear_to_cash: bool = False,
) -> RampPlan:
    """Compute the intended portfolio for a single rebalance.

    Args:
        as_of: planning timestamp.
        regime: regime label from MarketRegimeDetector.
        regime_confidence: detector confidence.
        regime_scores: all five regime scores (not just the winner).
        top_n: target position count for this regime.
        momentum_scores: pd.Series of momentum scores indexed by symbol.
                         Highest scores rank first; sorted descending.
        current_positions: dict of symbol -> current market value (USD).
        vix: current VIX level.
        spy_drawdown: trailing peak-to-current drawdown (negative number).
        max_capital_allocation: fraction of equity_base to deploy (default 1.0).
        diagnostics: extra fields recorded with the plan (passed to decision log).
        vix_threshold: VIX level above which crash protection fires (default 25).
        spy_dd_threshold: SPY drawdown below which crash protection fires
                          (default -0.05 = -5%).
        reduced_exposure: exposure multiplier when crash protection fires
                          (default 0.5).
        safe_mode: if True, return a plan that holds existing positions only.
                   No new entries, no exits.
        bear_to_cash: if True AND regime == "BEAR", target gross = 0.0 (all
                      existing positions become exits).

    Returns:
        RampPlan with target weights, exits, and diagnostics.
    """
    import pandas as pd  # local import to keep module-level deps minimal

    diag = dict(diagnostics or {})
    diag.setdefault("vix", vix)
    diag.setdefault("spy_dd", spy_drawdown)

    # Safe mode: hold existing positions, no new entries, no exits.
    if safe_mode:
        targets: Dict[str, RampTarget] = {}
        for sym in current_positions:
            targets[sym] = RampTarget(
                symbol=sym, target_weight=0.0, rank=None,
                regime=regime, reason="hold",
            )
        return RampPlan(
            as_of=as_of, regime=regime, regime_confidence=regime_confidence,
            regime_scores=regime_scores, exposure_pct=0.0, top_n=top_n,
            targets=targets, exits={}, diagnostics=diag,
        )

    # BEAR_TO_CASH variant: target gross = 0. Exit everything.
    if bear_to_cash and regime == "BEAR":
        exits = {sym: "bear_to_cash" for sym in current_positions}
        return RampPlan(
            as_of=as_of, regime=regime, regime_confidence=regime_confidence,
            regime_scores=regime_scores, exposure_pct=0.0, top_n=top_n,
            targets={}, exits=exits, diagnostics=diag,
        )

    # Crash protection
    crash_triggered = (vix > vix_threshold) or (spy_drawdown < spy_dd_threshold)
    exposure_pct = reduced_exposure if crash_triggered else 1.0
    diag["crash_triggered"] = crash_triggered

    # Target weight per position
    per_position_weight = max_capital_allocation * exposure_pct / top_n

    # Select top_n symbols by momentum
    ranked = momentum_scores.dropna().sort_values(ascending=False)
    top_symbols = list(ranked.head(top_n).index)
    rank_map = {sym: i + 1 for i, sym in enumerate(top_symbols)}

    # Build targets for top_n
    targets = {}
    for sym in top_symbols:
        reason = "hold" if sym in current_positions else "new_entry"
        targets[sym] = RampTarget(
            symbol=sym, target_weight=per_position_weight,
            rank=rank_map[sym], regime=regime, reason=reason,
        )

    # Existing positions not in top_n -> exit
    exits = {
        sym: "dropped_from_top_n"
        for sym in current_positions
        if sym not in targets
    }

    return RampPlan(
        as_of=as_of, regime=regime, regime_confidence=regime_confidence,
        regime_scores=regime_scores, exposure_pct=exposure_pct, top_n=top_n,
        targets=targets, exits=exits, diagnostics=diag,
    )
