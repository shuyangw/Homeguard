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

from src.research.ramp_phase4.engine import (
    _engine_pre_variant_update,
    _engine_pre_variant_update_soft_bear,
)
from src.research.ramp_phase4.filters import rank_buffer, min_hold
from src.research.ramp_phase4.plans import PLAN_CASH_BEAR_SOFT, _SentinelPlan
from src.strategies.advanced.market_regime_detector import (
    DataInsufficientError, MarketRegimeDetector,
)
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


def _compute_plan_from_panel(t: datetime, panel: pd.DataFrame, return_momentum: bool = False):
    """Wrap compute_plan with the parameters derivable from the panel and detector.

    Returns a RampPlan, or (RampPlan, momentum_scores) if return_momentum=True.
    The momentum_scores series carries ALL universe symbols ranked by score
    (highest first); Wave 1 filters need this to compute rank for held names
    that fell outside top_n.
    """
    spy = panel['SPY'].dropna()
    vix = panel['VIX'].dropna()
    if t not in spy.index or t not in vix.index:
        # No data at t; safe-mode.
        return (None, None) if return_momentum else None
    spy_slice = spy.loc[:t]
    vix_slice = vix.loc[:t]
    if len(spy_slice) < 252 or len(vix_slice) < 252:
        return (None, None) if return_momentum else None

    # Run detector to populate last_regime_scores + classify.
    spy_df = pd.DataFrame({
        'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
        'volume': 1e6,
    })
    vix_df = pd.DataFrame({'close': vix_slice})
    try:
        regime, confidence = _DETECTOR.classify_regime(spy_df, vix_df, t)
    except Exception:
        return (None, None) if return_momentum else None
    regime_scores = dict(_DETECTOR.last_regime_scores or {})

    # Momentum from universe prices excluding SPY/VIX.
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    prices_slice = panel.loc[:t, universe_cols]
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = REGIME_PARAMS.get(regime, REGIME_PARAMS.get('SIDEWAYS', {}))
    momentum = ramp.calculate_momentum_scores(prices_slice)
    if momentum is None or len(momentum) == 0:
        return (None, None) if return_momentum else None

    top_n = REGIME_PARAMS.get(regime, {}).get('top_n', 10)
    spy_dd = float((spy_slice.iloc[-1] / spy_slice.cummax().iloc[-1]) - 1.0)
    plan = compute_plan(
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
    if return_momentum:
        return plan, momentum
    return plan


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

    The universe ranking is built from the FULL momentum-scored universe (not just
    plan.targets), so previously-held names that fell out of top_n still have a
    rank the buffer can evaluate.
    """
    plan, momentum = _compute_plan_from_panel(t, panel, return_momentum=True)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    if not plan.targets:
        return {'__regime__': plan.regime}

    target_symbols = list(plan.targets.keys())
    proposed = {sym: 1.0 / plan.top_n for sym in target_symbols}

    # Full-universe ranking from sorted momentum scores (highest = rank 1).
    sorted_momentum = momentum.dropna().sort_values(ascending=False)
    ranking = pd.Series(
        range(1, len(sorted_momentum) + 1),
        index=sorted_momentum.index,
    )

    targets = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=plan.top_n // 2,
        universe_ranking=ranking,
        top_n=plan.top_n,
    )
    targets['__regime__'] = plan.regime
    return targets


def _variant_v05(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V05: V01 base + min-hold filter (protect positions younger than 5 trading days).

    crash_exit=False for the solo variant; V11 may override.
    """
    plan = _compute_plan_from_panel(t, panel)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    if not plan.targets:
        return {'__regime__': plan.regime}

    target_symbols = list(plan.targets.keys())
    proposed = {sym: 1.0 / plan.top_n for sym in target_symbols}

    targets = min_hold(
        proposed_targets=proposed,
        state=state,
        current_date=t,
        min_hold_days=5,
        crash_exit=False,
    )
    targets['__regime__'] = plan.regime
    return targets


def _variant_v11(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V11: V01 base + rank-buffer + min-hold (delta-threshold lives in cfg).

    Composition order: rank_buffer -> min_hold. The order matters because
    rank_buffer may add a new "buffered" symbol, after which min_hold can
    also see it. Reversing the order would let min_hold protect a name
    that rank_buffer was about to drop. cfg.delta_rebalance_pct=0.02 must
    be set at the CLI level (Task 11).
    """
    plan, momentum = _compute_plan_from_panel(t, panel, return_momentum=True)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    if not plan.targets:
        return {'__regime__': plan.regime}

    target_symbols = list(plan.targets.keys())
    proposed = {sym: 1.0 / plan.top_n for sym in target_symbols}

    # Full-universe ranking from sorted momentum scores.
    sorted_momentum = momentum.dropna().sort_values(ascending=False)
    ranking = pd.Series(
        range(1, len(sorted_momentum) + 1),
        index=sorted_momentum.index,
    )

    targets = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=plan.top_n // 2,
        universe_ranking=ranking,
        top_n=plan.top_n,
    )

    targets = min_hold(
        proposed_targets=targets,
        state=state,
        current_date=t,
        min_hold_days=5,
        crash_exit=False,
    )

    targets['__regime__'] = plan.regime
    return targets


def _variant_v12(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V12: V11 base + per-regime position override.

    Per spec rev4: the engine update happens INSIDE this variant (not in
    run_variant) because we need the regime to do the update, and the
    regime comes from V11's plan output. The update is pre-variant in
    the sense that it happens BEFORE the per-regime position decision.

    With cfg.min_regime_days > 0, the active regime used for the mode
    lookup is state.last_validated_regime (engine-managed). With
    min_regime_days == 0 (v12.0.0 default), it's the instantaneous regime.

    Position modes (cfg.regime_positions[regime]):
        - 'normal': pass V11 plan through unchanged
        - 'cash':   strip weights, return {'__regime__': regime}
        - 'hold':   return {'__regime__': 'SAFE_MODE'} (preserve positions)

    Any other value raises NotImplementedError (reserved for V13+ tickers).
    """
    plan = _variant_v11(t, state, panel, cfg)
    regime = plan['__regime__']

    _engine_pre_variant_update(state, regime, cfg.min_regime_days)

    if cfg.min_regime_days > 0:
        if state.last_validated_regime is None:
            active_mode = 'normal'
        else:
            active_mode = cfg.regime_positions.get(
                state.last_validated_regime, 'normal'
            )
    else:
        active_mode = cfg.regime_positions.get(regime, 'normal')

    if active_mode == 'normal':
        return plan
    elif active_mode == 'cash':
        return {'__regime__': regime}
    elif active_mode == 'hold':
        return {'__regime__': 'SAFE_MODE'}
    else:
        raise NotImplementedError(
            f"position_mode '{active_mode}' reserved for V13+"
        )


def _variant_v13_bear_invert(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V13-bear-invert: on detector BEAR, go to SPY 100%; otherwise V11.

    Tests the BEAR-as-buy hypothesis from V12's onset-alignment panel
    (gap_days mean -3.42 -- detector fires after SPY trough).

    NOT OOS in strict sense -- discovered from EXT-OOS inspection.
    """
    plan = _variant_v11(t, state, panel, cfg)
    regime = plan['__regime__']
    if regime == 'BEAR':
        return {'SPY': 1.0, '__regime__': 'BEAR'}
    return plan


def _variant_v14a_soft_bear_cash(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float] | _SentinelPlan:
    """V14a: BEAR_score Schmitt-trigger -> cash on enter.

    Pre-conditions: cfg.soft_bear_tau_in/tau_out loaded from
    config/research/v14_tau_constants.json by the orchestrator.

    Reads detector.last_regime_scores['BEAR'] AFTER making an explicit
    classify_regime call to guarantee freshness. Decouples from
    _compute_plan_from_panel ordering (V11's incidental detector call).

    State machine: _engine_pre_variant_update_soft_bear mutates
    state.in_bear_soft_mode based on Schmitt trigger.

    Action: if in_bear_soft_mode -> PLAN_CASH_BEAR_SOFT; else V11 plan.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    if panel is not None:
        spy_slice = panel['SPY'].dropna().loc[:t]
        vix_slice = panel['VIX'].dropna().loc[:t]
        if len(spy_slice) >= 252 and len(vix_slice) >= 252:
            spy_df = pd.DataFrame({
                'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
                'volume': 1e6,
            })
            vix_df = pd.DataFrame({'close': vix_slice})
        else:
            spy_df, vix_df = None, None
    else:
        spy_df, vix_df = None, None

    bear_score = None
    # Narrow except: detector data-insufficiency is expected during warm-up
    # or partial-coverage days (None inputs raise TypeError; short windows raise
    # DataInsufficientError). AssertionError from the freshness check below must
    # NOT be swallowed -- that would defeat the check's purpose.
    try:
        _DETECTOR.classify_regime(spy_df, vix_df, t)
    except (DataInsufficientError, TypeError, ValueError, KeyError):
        pass
    # Freshness check OUTSIDE the try -- AssertionError propagates as intended.
    # Stale-read guard: only consume scores if the detector classified THIS tick.
    if _DETECTOR.last_classification_timestamp == t and _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        return PLAN_CASH_BEAR_SOFT
    return plan_v11


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
    'V05': VariantSpec(
        id='V05',
        description='V01 + min hold (protect positions younger than 5 trading days)',
        plan_fn=_variant_v05,
    ),
    'V06': VariantSpec(
        id='V06',
        description='V01 + delta-rebalance threshold (cfg.delta_rebalance_pct must be set to 0.02 at CLI level)',
        plan_fn=_variant_v01,
    ),
    'V11': VariantSpec(
        id='V11',
        description='V01 + combined turnover-lite (V04 rank buffer + V05 min hold + V06 delta threshold via cfg)',
        plan_fn=_variant_v11,
    ),
    'V12': VariantSpec(
        id='V12',
        description='V11 + per-regime position override (BEAR -> cash default; min_regime_days=0 default; symmetric debouncing available)',
        plan_fn=_variant_v12,
    ),
    'V13-bear-invert': VariantSpec(
        id='V13-bear-invert',
        description='V11 + BEAR onset goes to SPY 100% (inverse of V12 BEAR-to-cash; tests BEAR-as-buy hypothesis)',
        plan_fn=_variant_v13_bear_invert,
    ),
    'V14a-soft-bear-cash': VariantSpec(
        id='V14a-soft-bear-cash',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> cash',
        plan_fn=_variant_v14a_soft_bear_cash,
    ),
}
