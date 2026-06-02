"""Variant registry + plan_fns for the Phase B harness.

V01: production REGIME_PARAMS, ignores crash exposure (matches existing reports).
V03: production REGIME_PARAMS, honors planner's crash exposure (target-weight-correct).

Both delegate to compute_plan from src.strategies.advanced.ramp_target_planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict
import numpy as np
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


def _variant_v14b_soft_bear_spy(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float]:
    """V14b: BEAR_score Schmitt-trigger -> SPY 100% on enter.

    Tests E1's BEAR-as-buy hypothesis under a working trigger (soft scores
    lead argmax by median 24 days at tau=0.3 per E3). V11's gross is
    constant 1.0 (V01 base ignores exposure_pct; filters renormalize),
    so 'V11 gross to SPY' simplifies to a fixed {SPY: 1.0} allocation.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    spy_slice = panel['SPY'].dropna().loc[:t] if panel is not None else None
    vix_slice = panel['VIX'].dropna().loc[:t] if panel is not None else None

    spy_df = None
    vix_df = None
    if spy_slice is not None and vix_slice is not None \
            and len(spy_slice) >= 252 and len(vix_slice) >= 252:
        spy_df = pd.DataFrame({
            'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
            'volume': 1e6,
        })
        vix_df = pd.DataFrame({'close': vix_slice})

    try:
        _DETECTOR.classify_regime(spy_df, vix_df, t)
    except (DataInsufficientError, TypeError, ValueError, KeyError):
        pass

    bear_score = None
    if _DETECTOR.last_classification_timestamp == t and _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        return {'SPY': 1.0, '__regime__': 'BEAR_SOFT_SPY'}
    return plan_v11


def _variant_v14c_soft_bear_dampen(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float]:
    """V14c: BEAR_score Schmitt-trigger -> V11 plan * dampen_factor on enter.

    Tests whether the right response to BEAR is risk reduction (not switch).
    cfg.soft_bear_dampen_factor (default 0.5) is the multiplier applied to
    all V11 symbol weights; __regime__ marker is preserved.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    spy_slice = panel['SPY'].dropna().loc[:t] if panel is not None else None
    vix_slice = panel['VIX'].dropna().loc[:t] if panel is not None else None

    spy_df = None
    vix_df = None
    if spy_slice is not None and vix_slice is not None \
            and len(spy_slice) >= 252 and len(vix_slice) >= 252:
        spy_df = pd.DataFrame({
            'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
            'volume': 1e6,
        })
        vix_df = pd.DataFrame({'close': vix_slice})

    try:
        _DETECTOR.classify_regime(spy_df, vix_df, t)
    except (DataInsufficientError, TypeError, ValueError, KeyError):
        pass

    bear_score = None
    if _DETECTOR.last_classification_timestamp == t and _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        dampened: Dict[str, float] = {}
        for sym, w in plan_v11.items():
            if sym == '__regime__':
                continue
            dampened[sym] = float(w) * cfg.soft_bear_dampen_factor
        dampened['__regime__'] = 'BEAR_SOFT_DAMPEN'
        return dampened
    return plan_v11


def _compute_beta_residual_ranking(
    t: datetime,
    panel: pd.DataFrame,
    beta_window: int = 90,
    return_window: int = 21,
) -> "pd.Series | None":
    """Compute cross-sectional beta-residual momentum scores (vectorized).

    For each universe symbol at date t:
      1. Estimate OLS beta to SPY over the trailing `beta_window` days.
      2. Residual momentum score = stock_return_window - beta * SPY_return_window.
      3. Return a Series of scores (higher = better), index = symbol.

    Returns None if there is insufficient history or SPY data is unavailable.

    Rationale (H6/H8 attack): in BEAR, RAMP's standard momentum ranking selects
    high-beta names (SMCI/ENPH/MU) that appeared strong because the market was
    rising, not due to idiosyncratic strength. Residualizing removes exactly the
    beta-driven component so the ranking captures genuine stock-level alpha.

    Design choice: beta_window=90 (midpoint of the 60-126d spec range) is fixed,
    not swept, to avoid a parameter-search trial against the experiment budget.

    Performance: uses vectorized numpy matrix operations over all symbols at once
    instead of a per-symbol Python loop. For 488 symbols at 90-day window, this
    reduces per-day cost from ~43k Python OLS iterations to a single matrix op.
    """
    spy = panel['SPY'].dropna()
    if t not in spy.index:
        return None
    spy_slice = spy.loc[:t]
    min_history = beta_window + return_window + 1
    if len(spy_slice) < min_history:
        return None

    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    if not universe_cols:
        return None

    # Slice the price matrix once for the window we need.
    needed_rows = beta_window + return_window + 2  # +2 for pct_change NaN drop
    prices_matrix = panel.loc[:t, universe_cols].iloc[-needed_rows:]
    spy_matrix = spy_slice.iloc[-needed_rows:]

    # Pct-change returns (drop first NaN row). fill_method=None: do not forward-fill NaN
    # gaps (consistent with engine's NaN -> forced-exit treatment).
    stock_rets = prices_matrix.pct_change(fill_method=None).iloc[1:]  # (beta_window+return_window+1, n_syms)
    spy_rets = spy_matrix.pct_change(fill_method=None).iloc[1:]       # same length

    if len(stock_rets) < beta_window + return_window:
        return None

    # Beta regression window: last `beta_window` rows of returns.
    x = spy_rets.iloc[-beta_window:].values.reshape(-1)       # (beta_window,)
    Y = stock_rets.iloc[-beta_window:].values                  # (beta_window, n_syms)

    # Valid columns: no NaN in beta window.
    valid_mask = ~np.isnan(Y).any(axis=0)
    Y_valid = Y[:, valid_mask]
    cols_valid = [c for c, v in zip(universe_cols, valid_mask) if v]

    if Y_valid.shape[1] == 0:
        return None

    # Vectorized OLS: beta_i = cov(x, Y_i) / var(x).
    x_centered = x - x.mean()
    x_var = float((x_centered ** 2).mean())
    if x_var < 1e-12:
        betas = np.zeros(Y_valid.shape[1])
    else:
        Y_centered = Y_valid - Y_valid.mean(axis=0, keepdims=True)
        betas = (x_centered @ Y_centered) / (len(x) * x_var)

    # 21-day returns: (last_price / price_21d_ago) - 1.
    prices_full = prices_matrix[cols_valid].values  # (needed_rows, n_valid)
    spy_full = spy_matrix.values                    # (needed_rows,)

    # Need return_window+1 price points from the end.
    if prices_full.shape[0] < return_window + 1:
        return None
    p_end = prices_full[-1, :]
    p_start = prices_full[-(return_window + 1), :]
    spy_21d_return = float(spy_full[-1] / spy_full[-(return_window + 1)] - 1.0)

    # Any stock where start price is NaN or zero -> exclude.
    start_valid = (p_start > 0) & ~np.isnan(p_start) & ~np.isnan(p_end)
    stock_21d = np.where(start_valid, p_end / p_start - 1.0, np.nan)
    residual_scores = stock_21d - betas * spy_21d_return

    result = pd.Series(
        residual_scores,
        index=cols_valid,
    ).dropna()
    if len(result) == 0:
        return None
    return result


def _variant_v31(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V31: V11 structure with beta-residual momentum ranking.

    Replaces V11's raw momentum ranking with beta-residual momentum:
      residual_score = stock_21d_return - beta_90d * SPY_21d_return

    Everything else is identical to V11 (rank_buffer + min_hold + delta via cfg),
    so the ONLY difference vs V11 is the ranking signal. This isolates the signal
    effect for a clean vs-V11 read.

    Rationale: H6/H8 from the root-cause investigation showed that RAMP's BEAR
    selections were dominated by high-beta names (SMCI/ENPH/MU) that appeared
    strong only because the market was rising. Residualizing the momentum score
    removes that beta-driven component so the selection favors genuine
    idiosyncratic alpha.

    Beta window: 90 days (fixed; midpoint of 60-126d spec range; NOT swept to
    conserve the DSR trial budget).
    """
    # Compute beta-residual scores.
    residual_scores = _compute_beta_residual_ranking(t, panel)

    # Need regime and top_n from the detector -- use same detector call as V11.
    spy = panel['SPY'].dropna()
    vix = panel['VIX'].dropna()
    if t not in spy.index or t not in vix.index:
        return {'__regime__': 'SAFE_MODE'}
    spy_slice = spy.loc[:t]
    vix_slice = vix.loc[:t]
    if len(spy_slice) < 252 or len(vix_slice) < 252:
        return {'__regime__': 'SAFE_MODE'}

    spy_df = pd.DataFrame({
        'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
        'volume': 1e6,
    })
    vix_df = pd.DataFrame({'close': vix_slice})
    try:
        regime, _confidence = _DETECTOR.classify_regime(spy_df, vix_df, t)
    except Exception:
        return {'__regime__': 'SAFE_MODE'}

    if residual_scores is None or len(residual_scores) == 0:
        return {'__regime__': 'SAFE_MODE'}

    top_n = REGIME_PARAMS.get(regime, {}).get('top_n', 10)

    # Select top_n symbols by residual score (highest = best).
    ranked = residual_scores.dropna().sort_values(ascending=False)
    if len(ranked) == 0:
        return {'__regime__': regime}
    target_symbols = list(ranked.head(top_n).index)
    if not target_symbols:
        return {'__regime__': regime}

    proposed = {sym: 1.0 / top_n for sym in target_symbols}

    # Build universe ranking from residual scores (same shape expected by rank_buffer).
    sorted_residual = ranked
    ranking = pd.Series(
        range(1, len(sorted_residual) + 1),
        index=sorted_residual.index,
    )

    targets = rank_buffer(
        proposed_targets=proposed,
        state=state,
        buffer_size=top_n // 2,
        universe_ranking=ranking,
        top_n=top_n,
    )

    targets = min_hold(
        proposed_targets=targets,
        state=state,
        current_date=t,
        min_hold_days=5,
        crash_exit=False,
    )

    targets['__regime__'] = regime
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
    'V14b-soft-bear-spy': VariantSpec(
        id='V14b-soft-bear-spy',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> SPY 100%',
        plan_fn=_variant_v14b_soft_bear_spy,
    ),
    'V14c-soft-bear-dampen': VariantSpec(
        id='V14c-soft-bear-dampen',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> V11 plan * dampen_factor (default 0.5)',
        plan_fn=_variant_v14c_soft_bear_dampen,
    ),
    'V31': VariantSpec(
        id='V31',
        description=(
            'Beta-residual momentum: ranks stocks by (21d_return - beta_90d * SPY_21d_return); '
            'V11 scaffold (rank_buffer + min_hold + delta via cfg). '
            'Attacks H6/H8 -- high-beta lagged winners removed from BEAR selections.'
        ),
        plan_fn=_variant_v31,
    ),
}
