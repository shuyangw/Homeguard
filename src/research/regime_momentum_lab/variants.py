"""Variant registry + plan_fns for the Phase B harness.

prod_no_crash: production REGIME_PARAMS, ignores crash exposure (matches existing reports).
prod:          production REGIME_PARAMS, honors planner's crash exposure (target-weight-correct).
plain:         vanilla momentum, fixed V1_PARAMS, no regime overlay (regime recorded for forensics).
bear_cash:     prod but BEAR regime -> 100% cash (bear_to_cash=True).

Both prod and prod_no_crash delegate to compute_plan from
src.strategies.advanced.ramp_target_planner. Legacy aliases V01/V03/V0/V1/V8 resolve
via resolve() for backward-compatible report labels.
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
    aliases: tuple = ()


# Module-scoped singletons; constructing RAMPSignals is cheap with empty universe.
_DETECTOR = MarketRegimeDetector()

# Vanilla-momentum fixed params. Origin: ramp_root_cause_20260505.py (archived).
V1_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10}


def _compute_plan_from_panel(t: datetime, panel: pd.DataFrame, bear_to_cash: bool = False):
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
        bear_to_cash=bear_to_cash,
    )


def _variant_prod_no_crash(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """prod_no_crash: 'fresh portfolio every day' baseline.

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


def _variant_prod(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """prod: target-weight-correct production.

    Same selection as prod_no_crash but honors planner's exposure_pct
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


def _variant_bear_cash(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """prod, but BEAR regime -> 100% cash (bear_to_cash=True)."""
    plan = _compute_plan_from_panel(t, panel, bear_to_cash=True)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    targets = list(plan.targets.keys())
    if not targets:
        return {'__regime__': plan.regime}
    per_weight = float(plan.exposure_pct) / len(targets)
    out: Dict[str, float] = {sym: per_weight for sym in targets}
    out['__regime__'] = plan.regime
    return out


def _variant_plain(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """Vanilla momentum: fixed params, equal-weight top-N, full exposure.

    Regime is recorded for forensics but has zero effect on selection, sizing,
    or exposure (no overlay).
    """
    spy = panel['SPY'].dropna()
    vix = panel['VIX'].dropna()
    if t not in spy.index or t not in vix.index:
        return {'__regime__': 'SAFE_MODE'}
    spy_slice = spy.loc[:t]
    vix_slice = vix.loc[:t]
    if len(spy_slice) < 252 or len(vix_slice) < 252:
        return {'__regime__': 'SAFE_MODE'}
    spy_df = pd.DataFrame({'close': spy_slice, 'open': spy_slice, 'high': spy_slice,
                           'low': spy_slice, 'volume': 1e6})
    vix_df = pd.DataFrame({'close': vix_slice})
    try:
        regime, _ = _DETECTOR.classify_regime(spy_df, vix_df, t)
    except Exception:
        regime = 'SAFE_MODE'
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    prices_slice = panel.loc[:t, universe_cols]
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = V1_PARAMS
    momentum = ramp.calculate_momentum_scores(prices_slice)
    if momentum is None or len(momentum) == 0:
        return {'__regime__': regime}
    top = list(momentum.sort_values(ascending=False).index[:V1_PARAMS['top_n']])
    if not top:
        return {'__regime__': regime}
    per_weight = 1.0 / len(top)
    out: Dict[str, float] = {sym: per_weight for sym in top}
    out['__regime__'] = regime
    return out


REGISTRY: Dict[str, VariantSpec] = {
    'plain': VariantSpec(
        id='plain',
        description='Vanilla momentum, fixed params (pen_w=5.0, top_n=10), no regime overlay; '
                    'regime recorded for forensics only',
        plan_fn=_variant_plain,
        aliases=('V1',),
    ),
    'prod': VariantSpec(
        id='prod',
        description='Production RAMP: regime overlay + per-regime params + 0.5x crash multiplier',
        plan_fn=_variant_prod,
        aliases=('V03', 'V0'),
    ),
    'prod_no_crash': VariantSpec(
        id='prod_no_crash',
        description='Overlay + per-regime params, crash multiplier ignored (parity-test baseline)',
        plan_fn=_variant_prod_no_crash,
        aliases=('V01',),
    ),
    'bear_cash': VariantSpec(
        id='bear_cash',
        description='Production overlay but BEAR regime -> 100% cash (bear_to_cash=True)',
        plan_fn=_variant_bear_cash,
        aliases=('V8',),
    ),
}


_ALL_ALIASES = [a for s in REGISTRY.values() for a in s.aliases]
assert len(_ALL_ALIASES) == len(set(_ALL_ALIASES)), 'Duplicate alias in REGISTRY'


def resolve(name: str) -> VariantSpec:
    """Look up a variant by canonical id or any registered alias.

    Lets historical report labels (V0, V01, V03, V1, V8) keep resolving after
    the rename. Raises KeyError with the full id list on miss. Note: v11 is a
    toggle value, not a registry variant.
    """
    if name in REGISTRY:
        return REGISTRY[name]
    for spec in REGISTRY.values():
        if name in spec.aliases:
            return spec
    raise KeyError(
        f'Unknown variant {name!r}. Known ids: {sorted(REGISTRY)}; '
        f'aliases: {sorted(a for s in REGISTRY.values() for a in s.aliases)}')
