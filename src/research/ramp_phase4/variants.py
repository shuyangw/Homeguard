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


_V26_LAMBDA = 1.0         # Fixed 21d-vs-5d weight. NOT swept (conserves DSR budget).
_V26_WINSOR_SIGMA = 3.0   # Winsorization clip level in cross-sectional sigma units.
_V26_BOUNDED_THRESHOLD = 1.0  # V27 sub-refinement: bounded-penalty threshold (z5 units).
# V27 bounded-penalty variant: tested as an informational sensitivity within V26,
# NOT a separate registry entry (conserves DSR budget). To activate: set
# _V26_USE_BOUNDED_PENALTY = True in a one-off sensitivity run (see spec). Default False.
_V26_USE_BOUNDED_PENALTY = False

_V28_LAMBDA_REV = 0.1  # Fixed 5d reversal penalty weight. NOT swept (conserves DSR budget).


def _compute_v26_zscore_scores(
    t: datetime,
    panel: pd.DataFrame,
    lambda_: float = _V26_LAMBDA,
    winsor_sigma: float = _V26_WINSOR_SIGMA,
    use_bounded_penalty: bool = _V26_USE_BOUNDED_PENALTY,
    bounded_threshold: float = _V26_BOUNDED_THRESHOLD,
) -> "pd.Series | None":
    """Compute cross-sectional z-score normalized momentum scores (vectorized).

    Score per universe symbol = z21 - lambda * penalty_term

    where:
      z21 = cross-sectional zscore(trailing 21d return), winsorized at winsor_sigma
      z5  = cross-sectional zscore(trailing 5d return),  winsorized at winsor_sigma
      penalty_term (default): z5  -- symmetric reversal penalty
      penalty_term (bounded, V27 sub-refinement): max(0, z5 - bounded_threshold)
        This only penalizes names with STRONG recent short-term momentum (z5 > 1.0)
        and ignores mild short-term moves.

    Rationale: the production V11 signal (RAMPSignals.calculate_momentum_scores) uses
    raw 21d and 5d returns. When raw-return magnitudes differ across market regimes
    (e.g., BEAR days show tiny returns, STRONG_BULL shows large swings) the 5d
    reversal penalty can dominate or be negligible depending on scale. Cross-sectional
    z-scoring makes the 21d and 5d terms comparable regardless of regime volatility,
    so the lambda weight has stable economic meaning: lambda=1.0 means the two terms
    are equally weighted after normalization.

    Winsorization at +/-winsor_sigma (applied BEFORE z-scoring) prevents a single
    symbol with an extreme raw return (e.g., a 10-sigma gap up) from dominating the
    cross-sectional distribution and distorting all other symbols' z-scores.

    Parameters:
      lambda_: weight on the short-term reversal penalty term (FIXED at 1.0; NOT swept).
      winsor_sigma: raw-return winsorization level in cross-sectional sigma (FIXED at 3.0).
      use_bounded_penalty: if True, applies V27 sub-refinement (penalty = max(0, z5-threshold)).
      bounded_threshold: V27 threshold in z5 units (FIXED at 1.0; NOT swept).

    Returns a Series of scores (higher = better), index = symbol.
    Returns None if there is insufficient history (< 21+2 rows).
    """
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    if not universe_cols:
        return None
    if t not in panel.index:
        return None

    prices_slice = panel.loc[:t, universe_cols]
    min_rows = 21 + 2  # need 22 price points for a 21-day return
    if len(prices_slice) < min_rows:
        return None

    prices = prices_slice.apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    n_rows = prices.shape[0]

    def _safe_return(lookback: int) -> np.ndarray:
        """Trailing lookback-day return for each symbol; NaN if start price invalid."""
        if n_rows < lookback + 1:
            return np.full(prices.shape[1], np.nan)
        p_end = prices[-1, :]
        p_start = prices[-(lookback + 1), :]
        valid = (p_start > 0) & ~np.isnan(p_start) & ~np.isnan(p_end)
        return np.where(valid, p_end / p_start - 1.0, np.nan)

    def _cross_sectional_zscore_winsorized(raw_ret: np.ndarray) -> np.ndarray:
        """Winsorize raw returns at +/-winsor_sigma, then z-score cross-sectionally.

        Winsorization order: clip at +/-winsor_sigma of the UNCLIPPED distribution,
        then z-score the clipped values. This prevents extreme outliers from pulling
        the mean/std before clipping.
        """
        valid_mask = ~np.isnan(raw_ret)
        if valid_mask.sum() < 2:
            return raw_ret  # not enough points to z-score; return as-is (all NaN anyway)
        vals = raw_ret[valid_mask]
        mu = np.mean(vals)
        sigma = np.std(vals, ddof=1)
        if sigma < 1e-12:
            z = np.zeros_like(raw_ret)
            z[~valid_mask] = np.nan
            return z
        # Clip at +/-winsor_sigma of the original distribution.
        lo = mu - winsor_sigma * sigma
        hi = mu + winsor_sigma * sigma
        clipped = np.where(valid_mask, np.clip(raw_ret, lo, hi), np.nan)
        # Re-compute mean/std of clipped values for final z-score.
        clipped_valid = clipped[valid_mask]
        mu2 = np.mean(clipped_valid)
        sigma2 = np.std(clipped_valid, ddof=1)
        if sigma2 < 1e-12:
            z = np.zeros_like(clipped)
            z[~valid_mask] = np.nan
            return z
        z = np.where(valid_mask, (clipped - mu2) / sigma2, np.nan)
        return z

    ret_21d = _safe_return(21)
    ret_5d = _safe_return(5)

    z21 = _cross_sectional_zscore_winsorized(ret_21d)
    z5 = _cross_sectional_zscore_winsorized(ret_5d)

    if use_bounded_penalty:
        # V27 bounded-penalty sub-refinement: only penalize strong recent momentum.
        penalty = np.maximum(0.0, z5 - bounded_threshold)
    else:
        penalty = z5

    scores_arr = z21 - lambda_ * penalty

    result = pd.Series(scores_arr, index=universe_cols).dropna()
    if len(result) == 0:
        return None
    return result


def _variant_v26(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V26: V11 structure with z-score normalized momentum ranking.

    Replaces V11's raw momentum ranking with a cross-sectionally z-scored signal:
      score = z(21d_return) - lambda * z(5d_return)

    where each term is cross-sectionally z-scored (winsorized at 3 sigma before
    z-scoring) over the universe on each day. lambda=1.0 is FIXED (NOT swept).

    Everything else is identical to V11 (rank_buffer + min_hold + delta via cfg),
    so the ONLY difference vs V11 is the ranking signal. This isolates the signal
    effect for a clean vs-V11 read.

    Rationale: the production V11 signal uses raw returns. When raw-return magnitudes
    differ across market regimes (tiny BEAR returns vs large STRONG_BULL swings),
    the 5d reversal penalty can dominate or be negligible depending on scale.
    Cross-sectional z-scoring makes the two terms comparable: lambda=1.0 means
    equal weighting of normalized momentum and normalized reversal penalty, so
    the penalty has stable economic meaning regardless of regime volatility.

    V27 sub-refinement (bounded penalty) is available as an informational
    sensitivity toggle (_V26_USE_BOUNDED_PENALTY) -- it is NOT a separate registry
    entry and does NOT affect V26 defaults.

    lambda=1.0, winsor_sigma=3.0, and (if activated) bounded_threshold=1.0 are all
    FIXED and NOT grid-searched, to conserve the DSR trial budget.

    Minimum history: >= 21 trading days; insufficient history returns SAFE_MODE.
    """
    scores = _compute_v26_zscore_scores(t, panel)

    # Need regime and top_n from the detector -- same call pattern as V31/V28.
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

    if scores is None or len(scores) == 0:
        return {'__regime__': 'SAFE_MODE'}

    top_n = REGIME_PARAMS.get(regime, {}).get('top_n', 10)

    # Select top_n symbols by z-score score (highest = best).
    ranked = scores.dropna().sort_values(ascending=False)
    if len(ranked) == 0:
        return {'__regime__': regime}
    target_symbols = list(ranked.head(top_n).index)
    if not target_symbols:
        return {'__regime__': regime}

    proposed = {sym: 1.0 / top_n for sym in target_symbols}

    # Build universe ranking from z-score scores (same shape expected by rank_buffer).
    ranking = pd.Series(
        range(1, len(ranked) + 1),
        index=ranked.index,
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


def _compute_v28_multihorizon_scores(
    t: datetime,
    panel: pd.DataFrame,
    blend_21d: float = 0.5,
    blend_63d: float = 0.3,
    blend_126d: float = 0.2,
    lambda_rev: float = _V28_LAMBDA_REV,
) -> "pd.Series | None":
    """Compute multi-horizon momentum ensemble scores (vectorized).

    Score per universe symbol = blend_21d*ret_21d + blend_63d*ret_63d
                                + blend_126d*ret_126d - lambda_rev*ret_5d

    where ret_Nd is the trailing N-trading-day simple return from the close
    panel at date t. lambda_rev=0.1 is a fixed reversal penalty (not swept).

    Rationale (H2 attack): a more stable, multi-horizon signal may reduce
    reliance on regime gating. Higher base-rate than single-window momentum
    because the blend averages over transient signal noise.

    Blend weights are fixed (0.5/0.3/0.2) and not grid-searched.
    Requires >= 126 + 2 price points (i.e. >= 127 rows) for the longest
    horizon return. Returns None if insufficient history.
    """
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    if not universe_cols:
        return None
    if t not in panel.index:
        return None

    prices_slice = panel.loc[:t, universe_cols]
    min_rows = 126 + 2  # need 127 price points for a 126-day return
    if len(prices_slice) < min_rows:
        return None

    # Convert to float64 numpy array; apply(pd.to_numeric) coerces object/NA columns.
    prices = prices_slice.apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    n_rows = prices.shape[0]

    def _safe_return(lookback: int) -> np.ndarray:
        """Trailing lookback-day return for each symbol; NaN if start price invalid."""
        if n_rows < lookback + 1:
            return np.full(prices.shape[1], np.nan)
        p_end = prices[-1, :]
        p_start = prices[-(lookback + 1), :]
        valid = (p_start > 0) & ~np.isnan(p_start) & ~np.isnan(p_end)
        return np.where(valid, p_end / p_start - 1.0, np.nan)

    ret_5d = _safe_return(5)
    ret_21d = _safe_return(21)
    ret_63d = _safe_return(63)
    ret_126d = _safe_return(126)

    scores = (
        blend_21d * ret_21d
        + blend_63d * ret_63d
        + blend_126d * ret_126d
        - lambda_rev * ret_5d
    )

    result = pd.Series(scores, index=universe_cols).dropna()
    if len(result) == 0:
        return None
    return result


def _variant_v28(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V28: V11 structure with multi-horizon momentum ensemble ranking.

    Replaces V11's raw momentum ranking with a blended multi-horizon score:
      score = 0.5*ret_21d + 0.3*ret_63d + 0.2*ret_126d - 0.1*ret_5d

    Everything else is identical to V11 (rank_buffer + min_hold + delta via cfg),
    so the ONLY difference vs V11 is the ranking signal. This isolates the signal
    effect for a clean vs-V11 read.

    Blend weights (0.5/0.3/0.2) and reversal lambda (0.1) are FIXED and NOT
    grid-searched, to conserve the DSR trial budget.

    Rationale (H2 attack): attacks signal instability -- a more stable,
    multi-horizon signal may reduce reliance on regime gating. Higher base-rate
    than single-window momentum because the blend averages over transient noise.

    Minimum history: >= 126 trading days; insufficient history returns SAFE_MODE.
    """
    scores = _compute_v28_multihorizon_scores(t, panel)

    # Need regime and top_n from the detector -- same call pattern as V31.
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

    if scores is None or len(scores) == 0:
        return {'__regime__': 'SAFE_MODE'}

    top_n = REGIME_PARAMS.get(regime, {}).get('top_n', 10)

    # Select top_n symbols by blended score (highest = best).
    ranked = scores.dropna().sort_values(ascending=False)
    if len(ranked) == 0:
        return {'__regime__': regime}
    target_symbols = list(ranked.head(top_n).index)
    if not target_symbols:
        return {'__regime__': regime}

    proposed = {sym: 1.0 / top_n for sym in target_symbols}

    # Build universe ranking from multi-horizon scores (same shape expected by rank_buffer).
    ranking = pd.Series(
        range(1, len(ranked) + 1),
        index=ranked.index,
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


# Fixed param set for V02: SIDEWAYS entry of REGIME_PARAMS.
# Rationale: SIDEWAYS (long_p=21, short_p=5, long_w=0.5, pen_w=2.0) is the most
# conservative regime param and represents a reasonable regime-agnostic baseline.
# top_n=10 is the production default for most regimes (WEAK_BULL, BEAR, UNPREDICTABLE).
# These values are FIXED and NOT varied across detected regimes.
_V02_FIXED_PARAMS = REGIME_PARAMS['SIDEWAYS'].copy()   # long_p=21, short_p=5, long_w=0.5, pen_w=2.0
_V02_FIXED_TOP_N = 10  # fixed; does NOT change with regime


def _variant_v02_v05(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V02+V05: vanilla momentum + min-hold, regime-free CONTROL variant.

    Selection: production RAMPSignals.calculate_momentum_scores with a SINGLE FIXED
    param set (_V02_FIXED_PARAMS = REGIME_PARAMS['SIDEWAYS']: long_p=21, short_p=5,
    long_w=0.5, pen_w=2.0). top_n=10 fixed. Equal-weight top_n. Exposure 1.0 always.

    NO regime apparatus: the detector is NOT called to alter params, top_n, or exposure.
    __regime__ is labelled 'VANILLA' to signal the variant is regime-free.

    Filter: min_hold(min_hold_days=5, crash_exit=False) applied AFTER selection.
    No rank_buffer (to keep the variant minimal: vanilla signal + only min-hold).
    delta_rebalance via cfg is respected (same as V11).

    Diagnostic purpose: if V02+V05 turnover lands near V11's 39%, the turnover
    blowup observed in V31/V28/V26 is signal-specific; if V02+V05 also blows up,
    the harness or filters are the cause.
    """
    # Insufficient-history path: RAMPSignals requires >= 100 rows.
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    if t not in panel.index:
        return {'__regime__': 'SAFE_MODE'}
    prices_slice = panel.loc[:t, universe_cols]
    if len(prices_slice) < 100:
        return {'__regime__': 'SAFE_MODE'}

    # Compute momentum with FIXED params; do NOT run the detector.
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = _V02_FIXED_PARAMS.copy()
    momentum = ramp.calculate_momentum_scores(prices_slice)
    if momentum is None or len(momentum) == 0:
        return {'__regime__': 'SAFE_MODE'}

    # Select top_n by momentum score (highest = best).
    ranked = momentum.dropna().sort_values(ascending=False)
    target_symbols = list(ranked.head(_V02_FIXED_TOP_N).index)
    if not target_symbols:
        return {'__regime__': 'VANILLA'}

    proposed = {sym: 1.0 / _V02_FIXED_TOP_N for sym in target_symbols}

    # Apply min_hold (V05 component): protect positions younger than 5 trading days.
    targets = min_hold(
        proposed_targets=proposed,
        state=state,
        current_date=t,
        min_hold_days=5,
        crash_exit=False,
    )
    targets['__regime__'] = 'VANILLA'
    return targets


# Fixed constants for V33-core.
# Selection signal: production RAMPSignals with SIDEWAYS params (same as V02+V05).
# top_n: 10 (fixed, same as V02+V05).
# Absolute-momentum horizons: 21d and 63d (both must be > 0 to qualify).
_V33_FIXED_PARAMS = REGIME_PARAMS['SIDEWAYS'].copy()
_V33_FIXED_TOP_N = 10
_V33_ABS_SHORT = 21   # short absolute-return horizon (trading days)
_V33_ABS_LONG = 63    # long absolute-return horizon (trading days)


def _variant_v33_core(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V33-core: absolute-momentum cash gate, detector-free, close-only.

    Selection: production RAMPSignals.calculate_momentum_scores with FIXED SIDEWAYS
    params (_V33_FIXED_PARAMS: long_p=21, short_p=5, long_w=0.5, pen_w=2.0).
    top_n=10 fixed. No regime detector call to alter params, top_n, or exposure.

    Absolute-momentum cash gate (the point of this variant):
      - Only BUY a symbol if ret_21d > 0 AND ret_63d > 0 (both absolute, not cross-
        sectional). Among qualifying symbols, select top_n by momentum score and
        equal-weight them with weight 1/top_n each (NOT 1/n_qualifying).
      - If fewer than top_n qualify, hold the REST IN CASH (gross < 1.0).
      - If zero qualify (broad downturn), return {'__regime__': 'ABSMOM'} -- fully cash.

    This provides detector-free crash protection: gross exposure shrinks endogenously
    when absolute price momentum turns broadly negative, with no regime-lag latency.

    Turnover control: min_hold(min_hold_days=5, crash_exit=False) after selection.
    __regime__ is labelled 'ABSMOM' when the variant is active (no detector involved).

    Insufficient history (need >= 63 rows for the long abs-return gate) -> SAFE_MODE.
    """
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    if not universe_cols:
        return {'__regime__': 'SAFE_MODE'}
    if t not in panel.index:
        return {'__regime__': 'SAFE_MODE'}

    prices_slice = panel.loc[:t, universe_cols]
    # Require at least 63+1 rows for the 63d absolute-return computation.
    if len(prices_slice) < _V33_ABS_LONG + 1:
        return {'__regime__': 'SAFE_MODE'}

    # Compute absolute returns on both horizons for the cash gate.
    prices_arr = prices_slice.apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
    n_rows = prices_arr.shape[0]

    def _abs_return(lookback: int) -> np.ndarray:
        if n_rows < lookback + 1:
            return np.full(len(universe_cols), np.nan)
        p_end = prices_arr[-1, :]
        p_start = prices_arr[-(lookback + 1), :]
        valid = (p_start > 0) & ~np.isnan(p_start) & ~np.isnan(p_end)
        return np.where(valid, p_end / p_start - 1.0, np.nan)

    ret_21d = _abs_return(_V33_ABS_SHORT)
    ret_63d = _abs_return(_V33_ABS_LONG)

    # Absolute-momentum gate: symbol qualifies only if BOTH horizons are positive.
    qualifying_mask = (ret_21d > 0) & (ret_63d > 0)
    qualifying_cols = [c for c, q in zip(universe_cols, qualifying_mask) if q]

    if not qualifying_cols:
        # Broad downturn: go fully to cash, no positions.
        return {'__regime__': 'ABSMOM'}

    # Among qualifying symbols, rank by momentum score (RAMPSignals SIDEWAYS params).
    qualifying_prices = prices_slice[qualifying_cols]
    ramp = RAMPSignals(symbols=qualifying_cols)
    ramp._current_params = _V33_FIXED_PARAMS.copy()
    momentum = ramp.calculate_momentum_scores(qualifying_prices)

    if momentum is None or len(momentum) == 0:
        return {'__regime__': 'ABSMOM'}

    ranked = momentum.dropna().sort_values(ascending=False)
    # Select top_n by momentum; take min(top_n, n_qualifying) to avoid over-selection.
    n_select = min(_V33_FIXED_TOP_N, len(ranked))
    target_symbols = list(ranked.head(n_select).index)

    if not target_symbols:
        return {'__regime__': 'ABSMOM'}

    # Equal-weight with 1/top_n each (NOT 1/n_qualifying) so gross < 1.0 when
    # n_qualifying < top_n, implementing the partial-exposure / cash-residual logic.
    per_weight = 1.0 / _V33_FIXED_TOP_N
    proposed = {sym: per_weight for sym in target_symbols}

    # Apply min_hold (5 days, crash_exit=False) for turnover control.
    targets = min_hold(
        proposed_targets=proposed,
        state=state,
        current_date=t,
        min_hold_days=5,
        crash_exit=False,
    )

    # After min_hold, re-normalize weights keeping the 1/top_n scale.
    # min_hold may add protected symbols; re-assign per_weight to maintain
    # equal-weighting at 1/top_n each (protected symbols also get 1/top_n,
    # which may push gross slightly above n_select/top_n but remains <= 1.0
    # as long as n_total_held <= top_n).
    n_total = len([k for k in targets if k != '__regime__'])
    if n_total > 0:
        # Reweight all held symbols at 1/top_n each (capped at 1.0 gross).
        targets = {sym: per_weight for sym in targets if sym != '__regime__'}
        gross = sum(targets.values())
        if gross > 1.0 + 1e-9:
            # Safety cap: normalize if somehow over 1.0 (shouldn't happen normally).
            norm_w = 1.0 / n_total
            targets = {sym: norm_w for sym in targets}

    targets['__regime__'] = 'ABSMOM'
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
    'V26': VariantSpec(
        id='V26',
        description=(
            'Z-score normalized score: score = z(21d_return) - lambda * z(5d_return); '
            'cross-sectional z-score per day, winsorized at 3 sigma. '
            'V11 scaffold (rank_buffer + min_hold + delta via cfg). '
            'lambda=1.0 fixed (not swept). Attacks scale instability of raw-return '
            'penalty term: z-scoring makes the 21d and 5d terms comparable across regimes.'
        ),
        plan_fn=_variant_v26,
    ),
    'V28': VariantSpec(
        id='V28',
        description=(
            'Multi-horizon momentum ensemble: '
            'score = 0.5*ret_21d + 0.3*ret_63d + 0.2*ret_126d - 0.1*ret_5d; '
            'V11 scaffold (rank_buffer + min_hold + delta via cfg). '
            'Attacks H2 (signal instability) -- multi-horizon blend reduces reliance '
            'on regime gating. Blend weights and reversal lambda are fixed (not swept).'
        ),
        plan_fn=_variant_v28,
    ),
    'V02+V05': VariantSpec(
        id='V02+V05',
        description=(
            'Vanilla momentum + min-hold, regime-free CONTROL variant. '
            'Uses production RAMPSignals.calculate_momentum_scores with a SINGLE FIXED '
            'param set (SIDEWAYS: long_p=21, short_p=5, long_w=0.5, pen_w=2.0); '
            'top_n=10 fixed; equal-weight; exposure 1.0 always. '
            'NO regime apparatus (detector NOT called to alter selection). '
            'min_hold(5 days, crash_exit=False) applied; no rank_buffer. '
            'CONTROL: turnover diagnostic vs V11 (39%) isolates whether blowup in '
            'V31/V28/V26 is signal-specific or harness-driven.'
        ),
        plan_fn=_variant_v02_v05,
    ),
    'V33-core': VariantSpec(
        id='V33-core',
        description=(
            'Absolute-momentum cash gate (detector-free, close-only). '
            'Selection: RAMPSignals SIDEWAYS params (long_p=21, short_p=5, long_w=0.5, '
            'pen_w=2.0), top_n=10 fixed; equal-weight at 1/top_n each. '
            'Cash gate: only buy symbols where ret_21d > 0 AND ret_63d > 0 (absolute); '
            'others excluded. Fewer than top_n qualifying -> gross < 1.0 (cash residual). '
            'Zero qualifying -> fully cash (__regime__=ABSMOM, no symbols). '
            'NO regime detector call. min_hold(5 days, crash_exit=False). '
            'Attacks H6/H8 endogenously -- de-risks without detector lag.'
        ),
        plan_fn=_variant_v33_core,
    ),
}
