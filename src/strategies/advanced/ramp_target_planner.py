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
