"""Variant registry + plan_fns for the Phase B harness.

V01: production REGIME_PARAMS, ignores crash exposure (matches existing reports).
V03: production REGIME_PARAMS, honors planner's crash exposure (target-weight-correct).

Both delegate to compute_plan from src.strategies.advanced.ramp_target_planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict
import pandas as pd

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
from src.strategies.advanced.ramp_strategy import RAMPSignals, REGIME_PARAMS
from src.strategies.advanced.ramp_target_planner import compute_plan


PlanFn = Callable[[datetime, object, pd.DataFrame, object], Dict[str, float]]


@dataclass(frozen=True)
class VariantSpec:
    id: str
    description: str
    plan_fn: PlanFn


# Module-scoped singletons; constructing RAMPSignals is cheap with empty universe.
_DETECTOR = MarketRegimeDetector()


def _compute_plan_from_panel(t: datetime, panel: pd.DataFrame) -> "RampPlan":
    """Wrap compute_plan with the parameters derivable from the panel and detector."""
    spy = panel['SPY'].dropna()
    vix = panel['VIX'].dropna()
    if t not in spy.index or t not in vix.index:
        # No data at t; safe-mode.
        return None
    spy_slice = spy.loc[:t]
    vix_slice = vix.loc[:t]
    if len(spy_slice) < 252 or len(vix_slice) < 252:
        return None

    # Run detector to populate last_regime_scores + classify.
    spy_df = pd.DataFrame({
        'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
        'volume': 1e6,
    })
    vix_df = pd.DataFrame({'close': vix_slice})
    try:
        regime, confidence = _DETECTOR.classify_regime(spy_df, vix_df, t)
    except Exception:
        return None
    regime_scores = dict(_DETECTOR.last_regime_scores or {})

    # Momentum from universe prices excluding SPY/VIX.
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    prices_slice = panel.loc[:t, universe_cols]
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = REGIME_PARAMS.get(regime, REGIME_PARAMS.get('SIDEWAYS', {}))
    momentum = ramp.calculate_momentum_scores(prices_slice)
    if momentum is None or len(momentum) == 0:
        return None

    top_n = REGIME_PARAMS.get(regime, {}).get('top_n', 10)
    spy_dd = float((spy_slice.iloc[-1] / spy_slice.cummax().iloc[-1]) - 1.0)
    return compute_plan(
        as_of=t,
        regime=regime,
        regime_confidence=confidence,
        regime_scores=regime_scores,
        top_n=top_n,
        momentum_scores=momentum,
        current_positions={},  # backtest is stateless from planner's perspective; engine tracks state
        vix=float(vix_slice.iloc[-1]),
        spy_drawdown=spy_dd,
    )


def _variant_v01(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V01: 'fresh portfolio every day' baseline.

    Uses planner output but IGNORES exposure_pct (sets gross to 1.0 always).
    """
    plan = _compute_plan_from_panel(t, panel)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    targets = list(plan.targets.keys())
    if not targets:
        return {'__regime__': plan.regime}
    per_weight = 1.0 / len(targets)  # ignore exposure_pct
    out: Dict[str, float] = {sym: per_weight for sym in targets}
    out['__regime__'] = plan.regime
    return out


def _variant_v03(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V03: target-weight-correct production.

    Same selection as V01 but honors planner's exposure_pct
    (1.0 normally, 0.5 in crash regimes).
    """
    plan = _compute_plan_from_panel(t, panel)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    targets = list(plan.targets.keys())
    if not targets:
        return {'__regime__': plan.regime}
    per_weight = float(plan.exposure_pct) / len(targets)
    out: Dict[str, float] = {sym: per_weight for sym in targets}
    out['__regime__'] = plan.regime
    return out


def _variant_v04(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V04: V01 base + rank-buffer filter (keep held names within top_n + buffer).

    buffer_size = top_n // 2 per regime (5 when top_n=10, 10 when top_n=20).
    """
    from src.research.ramp_phase4.filters import rank_buffer

    plan = _compute_plan_from_panel(t, panel)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    if not plan.targets:
        return {'__regime__': plan.regime}

    # Build the universe momentum ranking from the planner's target list.
    # plan.targets is dict[symbol -> RampTarget] ordered by rank (highest momentum first).
    target_symbols = list(plan.targets.keys())
    proposed = {sym: 1.0 / plan.top_n for sym in target_symbols}
    ranking = pd.Series({sym: i + 1 for i, sym in enumerate(target_symbols)})

    targets = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=plan.top_n // 2,
        universe_ranking=ranking,
        top_n=plan.top_n,
    )
    targets['__regime__'] = plan.regime
    return targets


REGISTRY: Dict[str, VariantSpec] = {
    'V01': VariantSpec(
        id='V01',
        description='Fresh portfolio every day; production REGIME_PARAMS; ignores crash exposure',
        plan_fn=_variant_v01,
    ),
    'V03': VariantSpec(
        id='V03',
        description='Target-weight-correct production; honors planner exposure_pct',
        plan_fn=_variant_v03,
    ),
    'V04': VariantSpec(
        id='V04',
        description='V01 + rank buffer (keep held names within top_n + buffer_size = top_n // 2)',
        plan_fn=_variant_v04,
    ),
}
