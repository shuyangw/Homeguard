#!/usr/bin/env python3
"""V14 factorial Phase D readiness orchestrator (Experiment 7).

V14 = V11 + Schmitt-trigger BEAR_score consumer, with three variants for the
on-soft-bear action:
  - V14a-soft-bear-cash:   go to cash
  - V14b-soft-bear-spy:    go to SPY 100%
  - V14c-soft-bear-dampen: scale V11 plan by dampen_factor (default 0.5)

Cloned from `ramp_phase4_v12c_readiness.py` with V14-specific adaptations:

  GATE-INFLUENCING (~30 runs):
    - V14 cost grid: {V14a, V14b, V14c} x {1, 5, 7.5, 10} bps
        x {near_close, one_day_lag} = 24 runs
    - Cross-variants at 5 bps near_close (PBO matrix companions):
        V01, V11, V12, V12c-cfg, V13-bear-invert = 5 runs
        (V14a/b/c at 5 bps near_close are reused from the cost grid)
    - V11 reference at 7.5 bps one_day_lag (Gate 5 no-regress baseline) = 1 run

  SENSITIVITY APPENDIX (INFORMATIONAL, NOT gate-influencing):
    - V14a tau-band sensitivity: 2 runs at 5 bps near_close
        (tau_out = TAU_IN - 0.05, TAU_IN - 0.15)
    - V14c dampen sensitivity:    2 runs at 5 bps near_close
        (dampen_factor = 0.25, 0.75)

Per-variant 5-gate evaluation (per spec rev2):
  1. PSR(variant @ 5bps near_close)         > 0.95   (vs SR=0)
  2. DSR(variant)                            > 0.95   (n_trials_project=36)
  3. PBO across {V01,V11,V12,V12c-cfg,V13-bear-invert,V14a,V14b,V14c}
                                             < 0.5    (CSCV s=16)
  4. lag delta                               <= max(0.2*nc, 0.1)
  5a. Sharpe(variant @ 7.5bps lag)           > 0.30
  5b. Sharpe(variant @ 7.5bps lag)           >= 0.9 * Sharpe(V11 @ 7.5bps lag)

Tier classification:
  - TIER 1: all structural pass + PSR/DSR pass + Sharpe(@5bps nc) lifts
            V11(@5bps nc) by at least 0.10
  - TIER 3: structural pass + PSR/DSR or lift fails
  - TIER 4: any structural gate fails

Diagnostic PBO (4 orthogonal variants {V01, V11, V12, V14a-soft-bear-cash}) is
reported alongside the gate PBO but is NOT gate-influencing.

Selection rule when multiple variants land in TIER 1 (pre-registered):
  Sharpe@5bps_nc desc -> lower PBO -> lower DSR penalty
  -> if still tied, recommend a run-off.

Tau values are loaded ONCE at module import from
`config/research/v14_tau_constants.json`. Do NOT hardcode them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.statistics.pbo import pbo
from src.backtesting.statistics.psr import psr
from src.backtesting.statistics.dsr import dsr, expected_max_sharpe
from src.research.ramp_phase4.config import (
    HarnessConfig,
    load_v14_tau_constants,
    _DEFAULT_REGIME_POSITIONS,
)
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import cagr, sharpe_ratio
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import get_logger


logger = get_logger(__name__)


# V14 tau loaded ONCE at module import (NOT hardcoded).
TAU_IN, TAU_OUT = load_v14_tau_constants()


# Variants under test (V14 factorial).
V14_VARIANTS: Tuple[str, ...] = (
    'V14a-soft-bear-cash',
    'V14b-soft-bear-spy',
    'V14c-soft-bear-dampen',
)

# Cost grid and timing modes for the per-variant grid.
COST_GRID_BPS: Tuple[float, ...] = (1.0, 5.0, 7.5, 10.0)
TIMING_MODES: Tuple[str, ...] = ('near_close', 'one_day_lag')

# Gate PBO: 8-variant set per spec rev2.
# V12c-cfg is V12 with regime_positions[UNPREDICTABLE]='cash' applied at cfg
# build time -- not a separate REGISTRY entry.
GATE_PBO_VARIANTS: Tuple[str, ...] = (
    'V01', 'V11', 'V12', 'V12c-cfg', 'V13-bear-invert',
    'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen',
)

# Diagnostic PBO: 4 orthogonal variants (reported alongside, NOT gate-influencing).
DIAGNOSTIC_PBO_VARIANTS: Tuple[str, ...] = (
    'V01', 'V11', 'V12', 'V14a-soft-bear-cash',
)

# Cross-variants ALSO run at 5 bps near_close for the gate PBO matrix.
# V14a/V14b/V14c at 5 bps near_close are reused from the cost grid.
CROSS_VARIANTS_5BPS_NC: Tuple[str, ...] = (
    'V01', 'V11', 'V12', 'V12c-cfg', 'V13-bear-invert',
)

# V12c synthesized via config override: V11 default + UNPREDICTABLE -> cash + BEAR -> cash.
# (BEAR -> cash is already the default in _DEFAULT_REGIME_POSITIONS.)
V12C_REGIME_POSITIONS: Dict[str, str] = {
    **dict(_DEFAULT_REGIME_POSITIONS),
    'UNPREDICTABLE': 'cash',
}


# V11-based variants (rank-buffer + min-hold + delta threshold) require
# cfg.delta_rebalance_pct = 0.02 per V11's docstring (Task 11 of Phase B).
DELTA_REBALANCE_PCT_BY_VARIANT: Dict[str, float] = {
    'V06':                       0.02,
    'V11':                       0.02,
    'V12':                       0.02,
    'V12c-cfg':                  0.02,
    'V13-bear-invert':           0.02,
    'V14a-soft-bear-cash':       0.02,
    'V14b-soft-bear-spy':        0.02,
    'V14c-soft-bear-dampen':     0.02,
}


# DSR n_trials_project audited count per spec rev2 honesty discipline:
#   V11+pre-V11 cohort:               22
#   V12 base + 4-grid sensitivity:     5
#   V12c base:                         1
#   V13 base:                          1
#   V14a, V14b, V14c base:             3
#   V14a tau-band sensitivity:         2
#   V14c dampen sensitivity:           2
#                                    ---
#                                     36
N_TRIALS_PROJECT = 36

# Cost tier used for PSR/DSR/PBO gates.
GATE_COST_BPS = 5.0
PBO_S = 16

# Gate thresholds (per spec rev2).
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
LAG_DEGRADATION_FRACTION = 0.2
LAG_DEGRADATION_FLOOR = 0.1
COST_FLOOR_SHARPE = 0.30
NO_REGRESS_VS_V11_FRACTION = 0.9
TIER_1_LIFT_THRESHOLD = 0.10

# V11 reference Sharpe at 5 bps near_close pulled from the V11 readiness doc.
# Used for the TIER 1 lift comparison.
V11_REF_SHARPE_AT_5BPS_NC = 0.5306  # placeholder reference; gate uses the freshly-run V11.

# Output report path.
OUTPUT_REPORT = Path('docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md')


@dataclass
class RunResult:
    label: str               # logical label (may differ from REGISTRY id, e.g. 'V12c-cfg')
    variant_id: str          # underlying REGISTRY id
    cost_bps: float
    timing_mode: str
    records: List[DailyRecord]
    sharpe: float
    cagr: float


def _git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='V14 factorial Phase D readiness orchestrator.')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke mode: run ONE V14a backtest @ 5bps near_close and exit (no report).')
    p.add_argument('--start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2017, 1, 1))
    p.add_argument('--end', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2026, 5, 16))
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path, default=OUTPUT_REPORT)
    return p.parse_args(argv)


def _registry_variant_id(label: str) -> str:
    """Map a logical label (e.g. 'V12c-cfg') to the underlying REGISTRY id."""
    if label == 'V12c-cfg':
        return 'V12'  # V12c-cfg uses V12 code + regime_positions override.
    return label


def _regime_positions_for(label: str) -> Optional[Dict[str, str]]:
    if label == 'V12c-cfg':
        return dict(V12C_REGIME_POSITIONS)
    return None


def _is_v14_variant(label: str) -> bool:
    return label in V14_VARIANTS


def _build_cfg(args, label: str, cost_bps: float, timing_mode: str,
               soft_bear_tau_in: float = None,
               soft_bear_tau_out: float = None,
               soft_bear_dampen_factor: float = None) -> HarnessConfig:
    """Build a HarnessConfig for a single (label, cost, timing, [tau, dampen]) cell."""
    delta_pct = DELTA_REBALANCE_PCT_BY_VARIANT.get(label, 0.0)
    kwargs = dict(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=delta_pct,
        soft_bear_tau_in=soft_bear_tau_in if soft_bear_tau_in is not None else TAU_IN,
        soft_bear_tau_out=soft_bear_tau_out if soft_bear_tau_out is not None else TAU_OUT,
    )
    if soft_bear_dampen_factor is not None:
        kwargs['soft_bear_dampen_factor'] = soft_bear_dampen_factor
    rps = _regime_positions_for(label)
    if rps is not None:
        kwargs['regime_positions'] = rps
    return HarnessConfig(**kwargs)


def _run(args, label: str, cost_bps: float, timing_mode: str,
         soft_bear_tau_in: float = None,
         soft_bear_tau_out: float = None,
         soft_bear_dampen_factor: float = None) -> RunResult:
    variant_id = _registry_variant_id(label)
    cfg = _build_cfg(args, label, cost_bps, timing_mode,
                     soft_bear_tau_in=soft_bear_tau_in,
                     soft_bear_tau_out=soft_bear_tau_out,
                     soft_bear_dampen_factor=soft_bear_dampen_factor)
    spec = REGISTRY[variant_id]
    logger.info(f'[+] Running {label} (REGISTRY={variant_id}) {timing_mode} cost={cost_bps}bps...')
    t0 = time.time()
    records = run_variant(cfg, spec)
    elapsed = time.time() - t0
    if not records:
        raise RuntimeError(f'{label} {timing_mode} {cost_bps}bps produced no records')
    rets = pd.Series([r.daily_return for r in records])
    eq = (1 + rets).cumprod() * args.initial_capital
    sr = sharpe_ratio(rets)
    cg = cagr(eq)
    logger.info(
        f'[+] {label} {timing_mode} {cost_bps}bps done in {elapsed:.1f}s: '
        f'Sharpe={sr:.3f} CAGR={cg:.2%} n_days={len(records)}'
    )
    return RunResult(label=label, variant_id=variant_id, cost_bps=cost_bps,
                     timing_mode=timing_mode, records=records, sharpe=sr, cagr=cg)


def _build_pbo_matrix(records_by_label: Dict[str, List[DailyRecord]],
                      labels: Tuple[str, ...]) -> np.ndarray:
    anchor = labels[0]
    dates_anchor = [r.date for r in records_by_label[anchor]]
    for v in labels[1:]:
        dates_v = [r.date for r in records_by_label[v]]
        if dates_v != dates_anchor:
            raise RuntimeError(
                f'Date mismatch: {anchor} has {len(dates_anchor)} dates, '
                f'{v} has {len(dates_v)}'
            )
    return np.column_stack([
        np.array([r.daily_return for r in records_by_label[v]], dtype=np.float64)
        for v in labels
    ])


def _compute_psr(records: List[DailyRecord]) -> Dict[str, float]:
    """PSR(SR_hat | SR_benchmark=0) in per-period (daily) units (methodology 2.2)."""
    rets = np.array([r.daily_return for r in records], dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 or rets.std() == 0:
        raise RuntimeError('Returns degenerate; cannot compute PSR.')
    sr_daily = float(rets.mean() / rets.std())
    sr_annual = sr_daily * float(np.sqrt(252))
    skew = float(np.mean(((rets - rets.mean()) / rets.std()) ** 3))
    pearson_kurt = float(np.mean(((rets - rets.mean()) / rets.std()) ** 4))
    psr_val = psr(sr_hat=sr_daily, sr_benchmark=0.0, n=len(rets),
                  skew=skew, kurt=pearson_kurt)
    return {
        'sr_daily': sr_daily,
        'sr_annual': sr_annual,
        'skew': skew,
        'pearson_kurt': pearson_kurt,
        'n_days': int(len(rets)),
        'psr': float(psr_val),
    }


def _evaluate_gates_per_variant(variant_id: str,
                                cost_grid: Dict[Tuple[str, float, str], RunResult],
                                v11_ref_lag_sharpe: float,
                                v11_5bps_nc_sharpe: float,
                                pbo_value: float,
                                dsr_value: float,
                                psr_value: float) -> Dict[str, object]:
    """5-gate evaluation per V14 variant. Returns a dict for report rendering."""
    nc_5 = cost_grid[(variant_id, 5.0, 'near_close')].sharpe
    lag_5 = cost_grid[(variant_id, 5.0, 'one_day_lag')].sharpe
    lag_75 = cost_grid[(variant_id, 7.5, 'one_day_lag')].sharpe

    nc_minus_lag = nc_5 - lag_5
    cap = max(LAG_DEGRADATION_FRACTION * abs(nc_5), LAG_DEGRADATION_FLOOR)
    lag_pass = nc_minus_lag <= cap

    psr_pass = psr_value > PSR_THRESHOLD
    dsr_pass = dsr_value > DSR_THRESHOLD
    pbo_pass = pbo_value < PBO_THRESHOLD
    cost_floor_pass = lag_75 > COST_FLOOR_SHARPE
    no_regress_pass = lag_75 >= NO_REGRESS_VS_V11_FRACTION * v11_ref_lag_sharpe

    lift_pass = nc_5 > v11_5bps_nc_sharpe + TIER_1_LIFT_THRESHOLD

    return {
        'variant': variant_id,
        'sharpe_5bps_nc': float(nc_5),
        'sharpe_5bps_lag': float(lag_5),
        'sharpe_75bps_lag': float(lag_75),
        'psr': float(psr_value),
        'psr_pass': bool(psr_pass),
        'dsr': float(dsr_value),
        'dsr_pass': bool(dsr_pass),
        'pbo': float(pbo_value),
        'pbo_pass': bool(pbo_pass),
        'nc_minus_lag': float(nc_minus_lag),
        'lag_cap': float(cap),
        'lag_pass': bool(lag_pass),
        'cost_floor_pass': bool(cost_floor_pass),
        'no_regress_pass': bool(no_regress_pass),
        'v11_ref_lag': float(v11_ref_lag_sharpe),
        'v11_5bps_nc': float(v11_5bps_nc_sharpe),
        'lift_pass': bool(lift_pass),
    }


def _classify_tier(gates: Dict[str, object]) -> str:
    """TIER 1 / TIER 3 / TIER 4 per spec rev2."""
    structural_pass = (
        gates['pbo_pass']
        and gates['lag_pass']
        and gates['cost_floor_pass']
        and gates['no_regress_pass']
    )
    if not structural_pass:
        return 'TIER 4'
    sig_pass = gates['psr_pass'] and gates['dsr_pass']
    if sig_pass and gates['lift_pass']:
        return 'TIER 1'
    return 'TIER 3'


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def _pf(b: bool) -> str:
    return 'PASS' if b else 'FAIL'


def _build_doc(
    args,
    sha: str,
    cost_grid: Dict[Tuple[str, float, str], RunResult],
    cross_runs: Dict[str, RunResult],
    v11_lag_7p5: RunResult,
    sensitivity_runs: Dict[str, RunResult],
    psr_by_variant: Dict[str, Dict[str, float]],
    dsr_by_variant: Dict[str, Dict[str, float]],
    pbo_gate_value: float,
    pbo_gate_matrix_shape: tuple,
    pbo_diagnostic_value: float,
    pbo_diagnostic_matrix_shape: tuple,
    gates_by_variant: Dict[str, Dict[str, object]],
    tiers_by_variant: Dict[str, str],
    selection: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append('# V14 factorial Phase D Readiness Report (Experiment 7)')
    lines.append('')
    lines.append(f'**Code commit**: {sha}')
    lines.append(f'**Data**: Alpaca SIP daily-aggregated, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Cost tier for PSR/DSR/PBO gates**: {GATE_COST_BPS} bps per side')
    lines.append(f'**n_trials_project**: {N_TRIALS_PROJECT}')
    lines.append(f'**Tau constants**: tau_in={TAU_IN:.6f}, tau_out={TAU_OUT:.6f} '
                 f'(loaded from config/research/v14_tau_constants.json)')
    lines.append('')
    lines.append(
        '**Variant definitions**: V14a/V14b/V14c are all V11 + a Schmitt-trigger '
        'BEAR_score consumer, differing in the on-soft-bear action:'
    )
    lines.append('- V14a-soft-bear-cash: go to cash')
    lines.append('- V14b-soft-bear-spy: go to SPY 100%')
    lines.append('- V14c-soft-bear-dampen: scale V11 plan by dampen_factor (default 0.5)')
    lines.append('')

    # ----- Summary -----
    lines.append('## Summary')
    lines.append('')
    lines.append('| Variant | Tier | Sharpe@5bps_nc | PSR | DSR | PBO | nc-lag | Sharpe@7.5bps_lag |')
    lines.append('|---|:---:|---:|---:|---:|---:|---:|---:|')
    for v in V14_VARIANTS:
        g = gates_by_variant[v]
        lines.append(
            f'| {v} | {tiers_by_variant[v]} | {g["sharpe_5bps_nc"]:.4f} | '
            f'{g["psr"]:.4f} | {g["dsr"]:.4f} | {g["pbo"]:.4f} | '
            f'{g["nc_minus_lag"]:+.4f} | {g["sharpe_75bps_lag"]:.4f} |'
        )
    lines.append('')
    if selection is not None:
        lines.append(f'**Selected for Phase D**: {selection}')
    else:
        lines.append('**Selected for Phase D**: NONE (no V14 variant landed in TIER 1)')
    lines.append('')
    lines.append(
        '_Selection rule when multiple TIER 1 candidates (pre-registered): '
        'Sharpe@5bps_nc desc -> lower PBO -> lower DSR penalty -> run-off if still tied._'
    )
    lines.append('')

    # ----- 5-gate verdict table per V14 variant -----
    for v in V14_VARIANTS:
        g = gates_by_variant[v]
        lines.append(f'## 5-gate verdict -- {v}')
        lines.append('')
        lines.append('| Gate | Result | Value | Threshold |')
        lines.append('|---|:---:|---:|---:|')
        lines.append(
            f'| 1. PSR({v} @ 5bps nc, vs SR=0) | {_pf(g["psr_pass"])} | '
            f'{g["psr"]:.4f} | > {PSR_THRESHOLD} |'
        )
        lines.append(
            f'| 2. DSR({v}, n_trials={N_TRIALS_PROJECT}) | {_pf(g["dsr_pass"])} | '
            f'{g["dsr"]:.4f} | > {DSR_THRESHOLD} |'
        )
        lines.append(
            f'| 3. PBO across 8 variants {GATE_PBO_VARIANTS} | {_pf(g["pbo_pass"])} | '
            f'{g["pbo"]:.4f} | < {PBO_THRESHOLD} |'
        )
        lines.append(
            f'| 4. Lag-degradation (5 bps) | {_pf(g["lag_pass"])} | '
            f'nc={g["sharpe_5bps_nc"]:.3f}, lag={g["sharpe_5bps_lag"]:.3f}, '
            f'nc-lag={g["nc_minus_lag"]:+.3f} | '
            f'<= max(0.2*|nc|, 0.1) = {g["lag_cap"]:.3f} |'
        )
        lines.append(
            f'| 5a. Sharpe({v} @ 7.5bps lag) > 0.30 | {_pf(g["cost_floor_pass"])} | '
            f'{g["sharpe_75bps_lag"]:.4f} | > {COST_FLOOR_SHARPE} |'
        )
        lines.append(
            f'| 5b. Sharpe({v}) >= 0.9 * Sharpe(V11) @ 7.5bps lag | '
            f'{_pf(g["no_regress_pass"])} | '
            f'{v}={g["sharpe_75bps_lag"]:.4f}, 0.9*V11='
            f'{NO_REGRESS_VS_V11_FRACTION * g["v11_ref_lag"]:.4f} '
            f'(V11={g["v11_ref_lag"]:.4f}) | >= {NO_REGRESS_VS_V11_FRACTION} * V11 |'
        )
        lines.append(
            f'| TIER 1 lift: Sharpe({v} @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10 | '
            f'{_pf(g["lift_pass"])} | {v}={g["sharpe_5bps_nc"]:.4f}, '
            f'V11={g["v11_5bps_nc"]:.4f}, delta={g["sharpe_5bps_nc"] - g["v11_5bps_nc"]:+.4f} | '
            f'> {TIER_1_LIFT_THRESHOLD} |'
        )
        lines.append('')
        lines.append(f'**{v} tier**: {tiers_by_variant[v]}')
        lines.append('')

    # ----- Cost grid per variant -----
    for v in V14_VARIANTS:
        lines.append(f'## Cost grid -- {v}')
        lines.append('')
        lines.append('| Cost bps | Mode | Sharpe | CAGR |')
        lines.append('|---:|:--|---:|---:|')
        for cb in COST_GRID_BPS:
            for mode in TIMING_MODES:
                rr = cost_grid[(v, cb, mode)]
                lines.append(f'| {cb:.1f} | {mode} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
        lines.append('')

    # ----- Cross-variants -----
    lines.append('## Cross-variants comparison (5 bps near_close)')
    lines.append('')
    lines.append('| Variant | Sharpe | CAGR |')
    lines.append('|---|---:|---:|')
    for v in CROSS_VARIANTS_5BPS_NC:
        rr = cross_runs[v]
        lines.append(f'| {v} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    for v in V14_VARIANTS:
        rr = cost_grid[(v, GATE_COST_BPS, 'near_close')]
        lines.append(f'| {v} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    # ----- PBO -----
    lines.append('## PBO')
    lines.append('')
    lines.append('### Gate PBO (8 variants -- gate-influencing)')
    lines.append('')
    lines.append(f'- Matrix shape: {pbo_gate_matrix_shape[0]} x {pbo_gate_matrix_shape[1]} '
                 f'({", ".join(GATE_PBO_VARIANTS)})')
    lines.append(f'- s (CSCV submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_gate_value:.4f}')
    if pbo_gate_value < PBO_THRESHOLD:
        lines.append(f'- Interpretation: {pbo_gate_value:.4f} < {PBO_THRESHOLD} = low overfitting evidence')
    else:
        lines.append(f'- Interpretation: {pbo_gate_value:.4f} >= {PBO_THRESHOLD} = elevated overfitting risk')
    lines.append('')
    lines.append('### Diagnostic PBO (4 orthogonal variants -- NOT gate-influencing)')
    lines.append('')
    lines.append(f'- Variant set: {", ".join(DIAGNOSTIC_PBO_VARIANTS)}')
    lines.append(f'- Matrix shape: {pbo_diagnostic_matrix_shape[0]} x {pbo_diagnostic_matrix_shape[1]}')
    lines.append(f'- s (CSCV submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_diagnostic_value:.4f}')
    pbo_divergence = abs(pbo_diagnostic_value - pbo_gate_value)
    if pbo_divergence > 0.2:
        lines.append(f'- **[!] PBO DIVERGENCE FLAG**: diagnostic PBO differs from gate PBO by '
                     f'{pbo_divergence:.4f} (> 0.20). Investigate variant correlation structure.')
    else:
        lines.append(f'- Divergence vs gate PBO: {pbo_divergence:.4f} (within 0.20 tolerance)')
    lines.append('')

    # ----- PSR / DSR detail per variant -----
    lines.append('## PSR / DSR detail per V14 variant')
    lines.append('')
    lines.append(
        '_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section '
        '2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values '
        'are reported for human narrative only._'
    )
    lines.append('')
    for v in V14_VARIANTS:
        pi = psr_by_variant[v]
        di = dsr_by_variant[v]
        lines.append(f'### {v}')
        lines.append('')
        lines.append('| Metric | Daily (formula input) | Annualized (narrative) |')
        lines.append('|---|---:|---:|')
        lines.append(f'| Observed Sharpe ({v}) | {pi["sr_daily"]:.6f} | {pi["sr_annual"]:.4f} |')
        lines.append(
            f'| Expected max under null (n_trials={N_TRIALS_PROJECT}) | '
            f'{di["expected_max_sharpe_daily"]:.6f} | '
            f'{di["expected_max_sharpe_annual"]:.4f} |'
        )
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---:|')
        lines.append(f'| PSR (vs SR=0) | {pi["psr"]:.4f} |')
        lines.append(f'| DSR probability (true SR > expected max) | {di["dsr"]:.4f} |')
        lines.append(f'| Trial Sharpes (annualized) | {di["trial_sharpes_str"]} |')
        lines.append(f'| sqrt(V[trial_sharpes]) (daily) | {di["v_sqrt_daily"]:.6f} |')
        lines.append(f'| Sample skewness | {pi["skew"]:.4f} |')
        lines.append(f'| Sample Pearson kurtosis | {pi["pearson_kurt"]:.4f} |')
        lines.append(f'| Sample size (days) | {pi["n_days"]} |')
        lines.append('')

    # ----- Sensitivity panel -----
    lines.append('## Sensitivity panels (INFORMATIONAL, NOT gate-influencing)')
    lines.append('')
    lines.append('_Per spec rev2 honesty discipline, these panels are NOT used in the gate '
                 'decision. They inform the post-deploy parameter-monitoring plan._')
    lines.append('')
    lines.append('| Panel key | Sharpe | CAGR |')
    lines.append('|---|---:|---:|')
    for key in sorted(sensitivity_runs.keys()):
        rr = sensitivity_runs[key]
        lines.append(f'| {key} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    # ----- V11 reference -----
    lines.append('## V11 reference values')
    lines.append('')
    lines.append('| Reference | Sharpe |')
    lines.append('|---|---:|')
    lines.append(f'| V11 @ 5 bps near_close (from cross-variants) | {cross_runs["V11"].sharpe:.4f} |')
    lines.append(f'| V11 @ 7.5 bps one_day_lag (Gate 5 baseline) | {v11_lag_7p5.sharpe:.4f} |')
    lines.append(f'| V11-readiness-doc Sharpe @ 7.5 bps lag | {V11_REF_SHARPE_AT_5BPS_NC:.4f} '
                 '(legacy reference; gate uses freshly-run V11) |')
    lines.append('')

    # ----- Methodology decisions -----
    lines.append('## Methodology decisions')
    lines.append('')
    lines.append(
        f'- n_trials_project = {N_TRIALS_PROJECT} (audited count per spec rev2 honesty '
        'discipline: V11+pre-V11 22 + V12+sensitivity 5 + V12c 1 + V13 1 + V14a/b/c 3 + '
        'V14a tau sens 2 + V14c dampen sens 2 = 36)'
    )
    lines.append(f'- PBO s = {PBO_S} (methodology Section 2.4 default)')
    lines.append(f'- Cost tier for PSR/DSR/PBO: {GATE_COST_BPS} bps per side')
    lines.append(
        '- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, '
        'trades executed at close T+1, MTM at close T+1.'
    )
    lines.append('- All metrics computed on net-of-cost daily returns.')
    lines.append('- Variants run at full window (start..end). Phase 4 is single-pass, '
                 'not walk-forward.')
    lines.append('- Gate 4 (directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- '
                 'lag > near_close is the safe direction and is not penalized.')
    lines.append('- Gate 5: both clauses required: '
                 'Sharpe(variant @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).')
    lines.append(
        '- Tier classification: TIER 1 (all 5 structural+sig gates pass AND Sharpe lift > '
        f'{TIER_1_LIFT_THRESHOLD} over V11 @ 5bps nc) / TIER 3 (structural pass, PSR+DSR '
        'or lift fails) / TIER 4 (any structural gate fails).'
    )
    lines.append(
        '- Tau constants loaded ONCE at module import from '
        '`config/research/v14_tau_constants.json` (NOT hardcoded).'
    )
    lines.append('- V12c-cfg is synthesized via `regime_positions[UNPREDICTABLE] = "cash"` on '
                 'the V12 REGISTRY plan_fn -- not a separate REGISTRY entry.')
    lines.append('- Diagnostic PBO (4 orthogonal variants) is reported alongside the gate PBO '
                 'but does NOT influence the gate verdict.')
    lines.append('- Sensitivity panels (V14a tau-band, V14c dampen) are INFORMATIONAL only.')
    lines.append('')

    # ----- Metadata -----
    lines.append('## Metadata')
    lines.append('')
    lines.append(f'- Git SHA: {sha}')
    lines.append(f'- Run datetime: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'- n_trials_project: {N_TRIALS_PROJECT}')
    lines.append(f'- Tau constants source: config/research/v14_tau_constants.json '
                 f'(tau_in={TAU_IN:.6f}, tau_out={TAU_OUT:.6f})')
    lines.append('- Total gate-influencing unique runs: 30 '
                 '(24 V14 cost grid + 5 cross-variants + 1 V11 ref)')
    lines.append(f'- Sensitivity panels: {len(sensitivity_runs)} '
                 '(informational only; V14a tau-band x2 + V14c dampen x2)')
    lines.append('')

    return '\n'.join(lines)


def _select_winner(tiers_by_variant: Dict[str, str],
                   gates_by_variant: Dict[str, Dict[str, object]]) -> Optional[str]:
    """Apply the pre-registered selection rule when >=1 variant lands in TIER 1.

    Rule: Sharpe@5bps_nc desc -> lower PBO -> lower DSR penalty (lower DSR is worse;
    rank by 1 - DSR ascending == DSR descending; tiebreak prefers higher DSR).
    If still tied -> return marker so the report flags a run-off.
    """
    tier_1 = [v for v in V14_VARIANTS if tiers_by_variant[v] == 'TIER 1']
    if not tier_1:
        return None
    # Sort by primary (Sharpe@5bps_nc desc), then PBO asc, then DSR desc.
    def _key(v: str):
        g = gates_by_variant[v]
        return (-g['sharpe_5bps_nc'], g['pbo'], -g['dsr'])
    tier_1_sorted = sorted(tier_1, key=_key)
    if len(tier_1_sorted) == 1:
        return tier_1_sorted[0]
    # Check if top 2 are tied on all three keys.
    top, runner_up = tier_1_sorted[0], tier_1_sorted[1]
    if _key(top) == _key(runner_up):
        return f'RUN-OFF RECOMMENDED among {tier_1_sorted}'
    return tier_1_sorted[0]


def _smoke_run(args) -> int:
    """Run ONE V14a backtest @ 5bps near_close and exit."""
    logger.info('[+] Smoke mode: one V14a-soft-bear-cash backtest @ 5bps near_close')
    t0 = time.time()
    rr = _run(args, 'V14a-soft-bear-cash', 5.0, 'near_close')
    elapsed = time.time() - t0
    logger.info(f'[+] Smoke complete in {elapsed:.1f}s. Sharpe={rr.sharpe:.4f}, '
                f'CAGR={rr.cagr:.2%}, n_days={len(rr.records)}')
    return 0


def main() -> int:
    args = _parse_args()

    if args.smoke:
        return _smoke_run(args)

    logger.info(
        f'[+] Starting V14 factorial readiness orchestrator (Experiment 7): '
        f'window {args.start.date()}..{args.end.date()}'
    )
    logger.info(f'[+] tau_in={TAU_IN:.6f} tau_out={TAU_OUT:.6f} '
                f'(loaded from config/research/v14_tau_constants.json)')
    t_start = time.time()

    # ----- Cost grid: 24 runs (3 V14 variants x 4 cost x 2 timing) -----
    cost_grid: Dict[Tuple[str, float, str], RunResult] = {}
    for variant_id in V14_VARIANTS:
        for cb in COST_GRID_BPS:
            for mode in TIMING_MODES:
                cost_grid[(variant_id, cb, mode)] = _run(args, variant_id, cb, mode)

    # ----- Cross-variants at 5 bps near_close (5 runs; V14 reused from grid) -----
    cross_runs: Dict[str, RunResult] = {}
    for v in CROSS_VARIANTS_5BPS_NC:
        cross_runs[v] = _run(args, v, GATE_COST_BPS, 'near_close')

    # ----- V11 reference for Gate 5 (1 run) -----
    v11_lag_7p5 = _run(args, 'V11', 7.5, 'one_day_lag')

    # ----- Sensitivity panels (4 runs; informational only) -----
    sensitivity_runs: Dict[str, RunResult] = {}
    # V14a tau-band sensitivity: tau_out = tau_in - delta for delta in {0.05, 0.15}
    for delta in (0.05, 0.15):
        tau_out_val = round(TAU_IN - delta, 6)
        if tau_out_val <= 0:
            continue
        key = f'V14a|tau_out={tau_out_val:.4f}'
        sensitivity_runs[key] = _run(
            args, 'V14a-soft-bear-cash', GATE_COST_BPS, 'near_close',
            soft_bear_tau_in=TAU_IN, soft_bear_tau_out=tau_out_val,
        )
    # V14c dampen sensitivity: dampen_factor in {0.25, 0.75}
    for dampen in (0.25, 0.75):
        key = f'V14c|dampen={dampen}'
        sensitivity_runs[key] = _run(
            args, 'V14c-soft-bear-dampen', GATE_COST_BPS, 'near_close',
            soft_bear_dampen_factor=dampen,
        )

    logger.info('[+] All backtests complete. Computing PSR / DSR / PBO ...')

    # ----- PBO -- gate (8 variants) -----
    gate_pbo_records: Dict[str, List[DailyRecord]] = {}
    for v in CROSS_VARIANTS_5BPS_NC:
        gate_pbo_records[v] = cross_runs[v].records
    for v in V14_VARIANTS:
        gate_pbo_records[v] = cost_grid[(v, GATE_COST_BPS, 'near_close')].records
    gate_matrix = _build_pbo_matrix(gate_pbo_records, GATE_PBO_VARIANTS)
    pbo_gate_value = pbo(gate_matrix, s=PBO_S)
    logger.info(f'[+] Gate PBO: {pbo_gate_value:.4f} (matrix {gate_matrix.shape})')

    # ----- PBO -- diagnostic (4 orthogonal) -----
    diag_pbo_records: Dict[str, List[DailyRecord]] = {}
    for v in DIAGNOSTIC_PBO_VARIANTS:
        if v in V14_VARIANTS:
            diag_pbo_records[v] = cost_grid[(v, GATE_COST_BPS, 'near_close')].records
        else:
            diag_pbo_records[v] = cross_runs[v].records
    diag_matrix = _build_pbo_matrix(diag_pbo_records, DIAGNOSTIC_PBO_VARIANTS)
    pbo_diagnostic_value = pbo(diag_matrix, s=PBO_S)
    logger.info(f'[+] Diagnostic PBO: {pbo_diagnostic_value:.4f} (matrix {diag_matrix.shape})')

    # ----- PSR / DSR per V14 variant -----
    psr_by_variant: Dict[str, Dict[str, float]] = {}
    dsr_by_variant: Dict[str, Dict[str, float]] = {}
    # Trial Sharpes (annualized) used as the DSR null distribution: cross-variants
    # + V14a/b/c at 5 bps near_close (the 8 GATE_PBO_VARIANTS columns).
    trial_labels = list(GATE_PBO_VARIANTS)
    trial_sharpes_annual = []
    for v in trial_labels:
        if v in V14_VARIANTS:
            trial_sharpes_annual.append(cost_grid[(v, GATE_COST_BPS, 'near_close')].sharpe)
        else:
            trial_sharpes_annual.append(cross_runs[v].sharpe)
    trial_sharpes_daily = [s / float(np.sqrt(252)) for s in trial_sharpes_annual]

    for v in V14_VARIANTS:
        rec = cost_grid[(v, GATE_COST_BPS, 'near_close')]
        pi = _compute_psr(rec.records)
        psr_by_variant[v] = pi
        logger.info(f'[+] PSR ({v} vs SR=0): {pi["psr"]:.4f}')

        sr_zero_daily = expected_max_sharpe(trial_sharpes_daily, N_TRIALS_PROJECT)
        sr_zero_annual = sr_zero_daily * float(np.sqrt(252))
        dsr_value = dsr(
            sr_hat=pi['sr_daily'],
            trial_sharpes=trial_sharpes_daily,
            n=pi['n_days'],
            skew=pi['skew'],
            kurt=pi['pearson_kurt'],
            n_trials_project=N_TRIALS_PROJECT,
        )
        v_sqrt = float(np.sqrt(np.var(trial_sharpes_daily, ddof=1)))
        dsr_by_variant[v] = {
            'dsr': float(dsr_value),
            'expected_max_sharpe_daily': float(sr_zero_daily),
            'expected_max_sharpe_annual': float(sr_zero_annual),
            'trial_sharpes_str': ', '.join(
                f'{lab}={s:.3f}' for lab, s in zip(trial_labels, trial_sharpes_annual)
            ),
            'v_sqrt_daily': v_sqrt,
        }
        logger.info(
            f'[+] DSR ({v}): prob={dsr_value:.4f} expected_max_annual={sr_zero_annual:.4f}'
        )

    # ----- Gate evaluation + tier classification per V14 variant -----
    v11_5bps_nc_sharpe = cross_runs['V11'].sharpe
    gates_by_variant: Dict[str, Dict[str, object]] = {}
    tiers_by_variant: Dict[str, str] = {}
    for v in V14_VARIANTS:
        gates_by_variant[v] = _evaluate_gates_per_variant(
            variant_id=v,
            cost_grid=cost_grid,
            v11_ref_lag_sharpe=v11_lag_7p5.sharpe,
            v11_5bps_nc_sharpe=v11_5bps_nc_sharpe,
            pbo_value=pbo_gate_value,
            dsr_value=dsr_by_variant[v]['dsr'],
            psr_value=psr_by_variant[v]['psr'],
        )
        tiers_by_variant[v] = _classify_tier(gates_by_variant[v])
        logger.info(f'[+] {v} tier: {tiers_by_variant[v]}')

    selection = _select_winner(tiers_by_variant, gates_by_variant)
    logger.info(f'[+] Selection: {selection}')

    # ----- Render report -----
    sha = _git_sha()
    md = _build_doc(
        args=args,
        sha=sha,
        cost_grid=cost_grid,
        cross_runs=cross_runs,
        v11_lag_7p5=v11_lag_7p5,
        sensitivity_runs=sensitivity_runs,
        psr_by_variant=psr_by_variant,
        dsr_by_variant=dsr_by_variant,
        pbo_gate_value=pbo_gate_value,
        pbo_gate_matrix_shape=gate_matrix.shape,
        pbo_diagnostic_value=pbo_diagnostic_value,
        pbo_diagnostic_matrix_shape=diag_matrix.shape,
        gates_by_variant=gates_by_variant,
        tiers_by_variant=tiers_by_variant,
        selection=selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    total_elapsed = time.time() - t_start
    logger.info(f'[+] Wrote {args.output}')
    logger.info(f'[+] Total wall-clock: {total_elapsed / 60:.2f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
