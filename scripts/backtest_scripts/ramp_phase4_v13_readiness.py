#!/usr/bin/env python3
"""V13-bear-invert readiness orchestrator (experiment 1).

V13 tests the BEAR-as-buy hypothesis discovered from V12's onset-alignment
panel (mean gap_days = -3.42 across 59 events 2017-2026 -- the detector
fires ~3.4 trading days AFTER the SPY drawdown trough). On BEAR days V13
allocates 100% SPY instead of cash; otherwise it defers to V11.

NOT OOS in strict sense -- V13 was discovered from inspection of EXT-OOS
data. DSR n_trials includes V13 (22 -> 23). Forward OOS validation is
still required before deploy regardless of verdict.

  GATE-INFLUENCING (15 runs):
    - V13 cost grid: {1, 5, 7.5, 10} bps x {near_close, one_day_lag} = 8 runs
    - Cross-variants at 5 bps near_close: V01, V04, V05, V06, V11, V12 = 6 runs
      (V13 @ 5 bps near_close already in the cost grid; no double count)
    - V11 @ 7.5 bps one_day_lag (Gate 5 reference) = 1 run

  NO sensitivity appendix (V13 is structurally different from V12; no
  UNPREDICTABLE-cash or debouncing analog).

Five gates (rev4 + rev4-followup):
  1. PSR(V13 @ 5bps near_close)            > 0.95          (vs SR=0)
  2. DSR(V13)                              > 0.95          (n_trials_project=23)
  3. PBO across 7 variants                 < 0.5           (CSCV s=16)
  4. lag delta (directional)               <= max(0.2*|nc|, 0.1)
  5a. Sharpe(V13 @ 7.5bps one_day_lag)     > 0.30
  5b. Sharpe(V13 @ 7.5bps one_day_lag)     >= 0.9 * Sharpe(V11 @ 7.5bps one_day_lag)

Plus a detector-onset alignment panel: for each BEAR onset in V13's record
stream, report the SPY-holding window and the gap-days between the actual
onset and the SPY drawdown trough within the +/-20-day window.
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
from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.data import load_universe_panel
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import cagr, sharpe_ratio
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import get_logger


logger = get_logger(__name__)


DELTA_REBALANCE_PCT_BY_VARIANT = {
    'V06': 0.02,
    'V11': 0.02,
    'V12': 0.02,
    'V13-bear-invert': 0.02,
}

# 7 variants used for PBO matrix at 5 bps near_close (V12 included so V13 vs
# V12 is a direct PBO neighbor; V12 was 6-variant, V13 is 7-variant -- this
# is documented as a methodology decision below).
CROSS_VARIANTS: Tuple[str, ...] = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12', 'V13-bear-invert')
GATE_TARGET = 'V13-bear-invert'

COST_GRID_BPS: Tuple[float, ...] = (1.0, 5.0, 7.5, 10.0)
GATE_COST_BPS = 5.0

# n_trials_project: hard-coded per spec (22 from V12 readiness + 1 for V13
# introduction). The orchestrator does NOT re-read experiments.duckdb so
# the verdict is reproducible from the spec; document the rationale.
N_TRIALS_PROJECT = 23
N_TRIALS_SOURCE = (
    'hard-coded: 22 (V12 readiness 2026-05-24, 4 from experiments.duckdb '
    '+ 18 from V12 readiness run) + 1 (V13 introduction)'
)

PBO_S = 16

# Gate thresholds.
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
LAG_DEGRADATION_FRACTION = 0.2   # relative cap (rev4)
LAG_DEGRADATION_FLOOR = 0.1      # absolute cap (rev4 floor)
COST_FLOOR_SHARPE = 0.3          # rev4-followup
COST_NO_REGRESS_FRACTION = 0.9   # V13 >= 0.9 * V11 at 7.5bps one_day_lag

# V11 reference Sharpe pulled from V11 readiness report (2026-05-23).
# Documentary fallback only; we ALSO re-run V11 @ 7.5bps one_day_lag inline.
V11_REF_SHARPE_AT_7P5BPS_LAG = 0.5306

# V11 references at 5 bps for fast comparison in the report header narrative.
V11_REF_SHARPE_AT_5BPS_NEAR_CLOSE = 0.528
V11_REF_SHARPE_AT_5BPS_LAG = 0.580
V12_REF_SHARPE_AT_5BPS_NEAR_CLOSE = 0.268
V12_REF_SHARPE_AT_5BPS_LAG = 0.665


@dataclass
class RunResult:
    variant_id: str
    cost_bps: float
    timing_mode: str
    records: List[DailyRecord]
    sharpe: float
    cagr: float
    label: str = ''  # optional human-readable tag


def _git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='V13-bear-invert readiness orchestrator (experiment 1).')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke mode: run ONE V13 backtest @ 5bps near_close and exit (no report).')
    p.add_argument('--start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2017, 1, 1))
    p.add_argument('--end', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2026, 5, 16))
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path,
                   default=Path('docs/reports/ramp/20260525_phase4_v13_readiness.md'))
    return p.parse_args(argv)


def _build_cfg(args, variant_id: str, cost_bps: float, timing_mode: str) -> HarnessConfig:
    delta_pct = DELTA_REBALANCE_PCT_BY_VARIANT.get(variant_id, 0.0)
    return HarnessConfig(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=delta_pct,
    )


def _run(args, variant_id: str, cost_bps: float, timing_mode: str,
         label: str = '') -> RunResult:
    cfg = _build_cfg(args, variant_id, cost_bps, timing_mode)
    spec = REGISTRY[variant_id]
    tag = label or variant_id
    logger.info(f'[+] Running {tag} {timing_mode} cost={cost_bps}bps...')
    t0 = time.time()
    records = run_variant(cfg, spec)
    elapsed = time.time() - t0
    if not records:
        raise RuntimeError(f'{tag} {timing_mode} {cost_bps}bps produced no records')
    rets = pd.Series([r.daily_return for r in records])
    eq = (1 + rets).cumprod() * args.initial_capital
    sr = sharpe_ratio(rets)
    cg = cagr(eq)
    logger.info(
        f'[+] {tag} {timing_mode} {cost_bps}bps done in {elapsed:.1f}s: '
        f'Sharpe={sr:.3f} CAGR={cg:.2%} n_days={len(records)}'
    )
    return RunResult(variant_id=variant_id, cost_bps=cost_bps, timing_mode=timing_mode,
                     records=records, sharpe=sr, cagr=cg, label=label or variant_id)


def _build_pbo_matrix(records_by_variant: Dict[str, List[DailyRecord]]) -> np.ndarray:
    anchor = CROSS_VARIANTS[0]
    dates_anchor = [r.date for r in records_by_variant[anchor]]
    for v in CROSS_VARIANTS[1:]:
        dates_v = [r.date for r in records_by_variant[v]]
        if dates_v != dates_anchor:
            raise RuntimeError(
                f'Date mismatch: {anchor} has {len(dates_anchor)} dates, '
                f'{v} has {len(dates_v)}'
            )
    return np.column_stack([
        np.array([r.daily_return for r in records_by_variant[v]], dtype=np.float64)
        for v in CROSS_VARIANTS
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


def _detector_onset_alignment(records: List[DailyRecord],
                              panel: pd.DataFrame) -> Dict[str, object]:
    """For each BEAR onset, report SPY-holding window + lag-tax proxy.

    Same panel structure as V12 readiness, but the "cash window" semantics
    are inverted: for V13, the contiguous BEAR window is the period during
    which V13 holds 100% SPY. We measure SPY return during that window
    (V13's realized return on the BEAR branch, before V11 turnover-lite
    transaction cost is netted; the gate-influencing daily returns already
    net this).
    """
    if not records or 'SPY' not in panel.columns:
        return {'events': [], 'aggregate_gap_days': 0.0, 'aggregate_spy_return': 0.0}

    bear_onsets: List[pd.Timestamp] = []
    for i in range(1, len(records)):
        if records[i].regime == 'BEAR' and records[i - 1].regime != 'BEAR':
            bear_onsets.append(pd.Timestamp(records[i].date))

    spy = panel['SPY'].dropna()
    record_dates = [pd.Timestamp(r.date) for r in records]

    events: List[Dict[str, object]] = []
    for onset in bear_onsets:
        try:
            idx_onset = record_dates.index(onset)
        except ValueError:
            continue

        # BEAR window: contiguous days from onset where regime stays BEAR.
        bear_start = onset
        bear_end = onset
        for j in range(idx_onset, len(records)):
            if records[j].regime == 'BEAR':
                bear_end = pd.Timestamp(records[j].date)
            else:
                break

        win_lo = onset - pd.Timedelta(days=20)
        win_hi = onset + pd.Timedelta(days=30)
        spy_window = spy.loc[(spy.index >= win_lo) & (spy.index <= win_hi)]

        # SPY return during the BEAR window: the return V13 captured by
        # being long SPY (gross of any V11-side turnover during the window).
        spy_bear_window = spy.loc[(spy.index >= bear_start) & (spy.index <= bear_end)]
        if len(spy_bear_window) >= 2:
            spy_return = float(spy_bear_window.iloc[-1] / spy_bear_window.iloc[0] - 1.0)
        else:
            spy_return = 0.0

        gap_days = 0
        trough_date: Optional[pd.Timestamp] = None
        if len(spy_window) > 0:
            trough_date = spy_window.idxmin()
            mask = (spy.index >= min(onset, trough_date)) & (spy.index <= max(onset, trough_date))
            n_between = int(mask.sum())
            gap_days = (n_between - 1) if onset >= trough_date else -(n_between - 1)

        events.append({
            'onset': str(onset.date()),
            'window_start': str(spy_window.index[0].date()) if len(spy_window) else None,
            'window_end': str(spy_window.index[-1].date()) if len(spy_window) else None,
            'bear_start': str(bear_start.date()),
            'bear_end': str(bear_end.date()),
            'bear_days': int((bear_end - bear_start).days) + 1,
            'spy_trough_date': str(trough_date.date()) if trough_date is not None else None,
            'gap_days': int(gap_days),
            'spy_return': float(spy_return),
        })

    n = len(events)
    return {
        'events': events,
        'aggregate_gap_days': float(sum(e['gap_days'] for e in events) / n) if n else 0.0,
        'aggregate_spy_return': float(sum(e['spy_return'] for e in events) / n) if n else 0.0,
    }


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def _build_doc(
    args,
    sha: str,
    cost_grid_results: Dict[Tuple[float, str], RunResult],
    cross_runs_at_gate_cost: Dict[str, RunResult],
    v11_lag_7p5: RunResult,
    psr_info: Dict[str, float],
    dsr_info: Dict[str, float],
    pbo_value: float,
    matrix_shape: tuple,
    alignment: Dict[str, object],
    n_trials_used: int,
    n_trials_source: str,
) -> str:
    nc_v13_5 = cost_grid_results[(GATE_COST_BPS, 'near_close')].sharpe
    lag_v13_5 = cost_grid_results[(GATE_COST_BPS, 'one_day_lag')].sharpe
    v13_lag_7p5 = cost_grid_results[(7.5, 'one_day_lag')].sharpe

    # Gate 1: PSR
    psr_value = psr_info['psr']
    psr_pass = psr_value > PSR_THRESHOLD

    # Gate 2: DSR
    dsr_value = dsr_info['dsr']
    dsr_pass = dsr_value > DSR_THRESHOLD

    # Gate 3: PBO
    pbo_pass = pbo_value < PBO_THRESHOLD

    # Gate 4: directional (rev4) - same as V12 readiness.
    nc_minus_lag = nc_v13_5 - lag_v13_5
    cap = max(LAG_DEGRADATION_FRACTION * abs(nc_v13_5), LAG_DEGRADATION_FLOOR)
    lag_pass = nc_minus_lag <= cap

    # Gate 5: cost-floor + no-regress (both clauses)
    v11_ref = v11_lag_7p5.sharpe
    cost_floor_pass = v13_lag_7p5 > COST_FLOOR_SHARPE
    no_regress_pass = v13_lag_7p5 >= COST_NO_REGRESS_FRACTION * v11_ref
    cost_pass = cost_floor_pass and no_regress_pass

    # Tier classification per V13 spec (different from V12 verdict bands).
    # TIER 1: passes all 5 gates AND nc Sharpe materially beats V11.
    # TIER 3: structural pass + absolute-significance fail (similar to V12).
    # TIER 4: any structural gate fails.
    v11_nc_ref = V11_REF_SHARPE_AT_5BPS_NEAR_CLOSE
    sharpe_lift_vs_v11 = nc_v13_5 - v11_nc_ref
    sharpe_lift_threshold = 0.10
    sharpe_lift_pass = sharpe_lift_vs_v11 > sharpe_lift_threshold

    all_gates = [psr_pass, dsr_pass, pbo_pass, lag_pass, cost_pass]
    structural_gates = [pbo_pass, lag_pass, cost_pass]

    if all(all_gates) and sharpe_lift_pass:
        verdict_short = 'TIER 1'
        verdict = (
            'BEAR-as-buy is real and material -- V13 passes all 5 gates AND lifts '
            f'Sharpe vs V11 at 5bps near_close by {sharpe_lift_vs_v11:+.3f} '
            f'(threshold +{sharpe_lift_threshold:.2f}). WS-3 roadmap reframes around '
            'sign-inversion. Forward OOS validation still required (V13 was '
            'discovered from EXT-OOS inspection; NOT strict OOS).'
        )
    elif all(structural_gates) and not (psr_pass and dsr_pass):
        verdict_short = 'TIER 3'
        verdict = (
            'BEAR-as-buy is suggestive but not actionable -- structural gates '
            'PASS, absolute-significance gates FAIL (PSR/DSR). Effect size '
            'similar to V12 magnitude on the opposite side. Continue WS-3c '
            'roadmap. Forward OOS validation could change the verdict but '
            'paper deploy NOT warranted now.'
        )
    else:
        verdict_short = 'TIER 4'
        verdict = (
            'BEAR-as-buy is spurious -- one or more structural gates FAIL. '
            'Close V13; continue WS-3c roadmap.'
        )

    def _pf(b: bool) -> str:
        return 'PASS' if b else 'FAIL'

    lines: List[str] = []
    lines.append('# V13-bear-invert Readiness Report (Experiment 1)')
    lines.append('')
    lines.append(f'**Code commit**: {sha}')
    lines.append(f'**Data**: Alpaca SIP daily-aggregated, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Cost tier for PSR/DSR/PBO gates**: {GATE_COST_BPS} bps per side')
    lines.append(f'**n_trials_project**: {n_trials_used} ({n_trials_source})')
    lines.append('')
    lines.append(
        '> **Honesty discipline**: V13-bear-invert was discovered from inspection '
        "of V12's onset-alignment panel (mean gap_days = -3.42 across 59 events "
        '2017-2026 -- the detector fires AFTER the SPY drawdown trough). V13 '
        'inverts the sign of V12. This was discovered from EXT-OOS data and is '
        '**NOT OOS in the strict sense**; the same 2017-2026 window that '
        'motivated the hypothesis is the test window. Forward OOS validation '
        'is required before any paper deploy regardless of verdict. The DSR '
        'n_trials_project counter has been incremented (22 -> 23) to reflect '
        "V13's introduction."
    )
    lines.append('')

    lines.append('## Summary')
    lines.append('')
    lines.append('| Gate | Result | Value | Threshold |')
    lines.append('|---|:---:|---:|---:|')
    lines.append(
        f'| 1. PSR(V13 @ 5bps near_close, vs SR=0) | {_pf(psr_pass)} | '
        f'{psr_value:.4f} | > {PSR_THRESHOLD} |'
    )
    lines.append(
        f'| 2. DSR(V13, n_trials={n_trials_used}) | {_pf(dsr_pass)} | '
        f'{dsr_value:.4f} | > {DSR_THRESHOLD} |'
    )
    lines.append(
        f'| 3. PBO across {len(CROSS_VARIANTS)} variants | {_pf(pbo_pass)} | '
        f'{pbo_value:.4f} | < {PBO_THRESHOLD} |'
    )
    lines.append(
        f'| 4. Lag-degradation (5 bps, directional) | {_pf(lag_pass)} | '
        f'nc={nc_v13_5:.3f}, lag={lag_v13_5:.3f}, nc-lag={nc_minus_lag:+.3f} | '
        f'<= max(0.2*|nc|, 0.1) = {cap:.3f} |'
    )
    lines.append(
        f'| 5a. Sharpe(V13 @ 7.5bps lag) > 0.30 | {_pf(cost_floor_pass)} | '
        f'{v13_lag_7p5:.4f} | > {COST_FLOOR_SHARPE} |'
    )
    lines.append(
        f'| 5b. Sharpe(V13) >= 0.9 * Sharpe(V11) @ 7.5bps lag | {_pf(no_regress_pass)} | '
        f'V13={v13_lag_7p5:.4f}, 0.9*V11={COST_NO_REGRESS_FRACTION * v11_ref:.4f} '
        f'(V11={v11_ref:.4f}) | >= {COST_NO_REGRESS_FRACTION} * V11 |'
    )
    lines.append('')
    lines.append('### V13 vs V11 vs V12 -- direct head-to-head at 5 bps near_close')
    lines.append('')
    v12_inline = cross_runs_at_gate_cost.get('V12')
    v11_inline = cross_runs_at_gate_cost.get('V11')
    v12_inline_nc = v12_inline.sharpe if v12_inline else float('nan')
    v11_inline_nc = v11_inline.sharpe if v11_inline else float('nan')
    lines.append('| Variant | Sharpe @ 5bps near_close | Delta vs V11 |')
    lines.append('|---|---:|---:|')
    lines.append(f'| V11 (inline, baseline) | {v11_inline_nc:.4f} | 0.0000 |')
    lines.append(f'| V12 (BEAR -> cash, inline) | {v12_inline_nc:.4f} | {v12_inline_nc - v11_inline_nc:+.4f} |')
    lines.append(f'| **V13 (BEAR -> SPY 100%)** | **{nc_v13_5:.4f}** | **{nc_v13_5 - v11_inline_nc:+.4f}** |')
    lines.append('')
    lines.append(
        f'TIER 1 lift threshold (V13 vs V11 @ 5bps near_close): '
        f'+{sharpe_lift_threshold:.2f} required, observed '
        f'{sharpe_lift_vs_v11:+.4f} (vs documentary V11={v11_nc_ref:.3f}). '
        f'Lift gate: {"PASS" if sharpe_lift_pass else "FAIL"}.'
    )
    lines.append('')
    lines.append(f'**Overall verdict**: {verdict_short} -- {verdict}')
    lines.append('')

    lines.append('## PSR / DSR detail')
    lines.append('')
    lines.append(
        '_Units note: PSR and DSR formulas (Bailey-Lopez de Prado / methodology Section '
        '2.2-2.3) require **per-period (daily)** Sharpe with daily `n`. Annualized values '
        'are reported for human narrative only._'
    )
    lines.append('')
    lines.append('| Metric | Daily (formula input) | Annualized (narrative) |')
    lines.append('|---|---:|---:|')
    lines.append(f'| Observed Sharpe (V13) | {psr_info["sr_daily"]:.6f} | {psr_info["sr_annual"]:.4f} |')
    lines.append(
        f'| Expected max under null (n_trials={n_trials_used}) | '
        f'{dsr_info["expected_max_sharpe_daily"]:.6f} | '
        f'{dsr_info["expected_max_sharpe_annual"]:.4f} |'
    )
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    lines.append(f'| PSR (vs SR=0) | {psr_value:.4f} |')
    lines.append(f'| DSR probability (true SR > expected max) | {dsr_value:.4f} |')
    lines.append(f'| Trial Sharpes (annualized) | {dsr_info["trial_sharpes_str"]} |')
    lines.append(f'| sqrt(V[trial_sharpes]) (daily) | {dsr_info["v_sqrt_daily"]:.6f} |')
    lines.append(f'| Sample skewness | {psr_info["skew"]:.4f} |')
    lines.append(f'| Sample Pearson kurtosis | {psr_info["pearson_kurt"]:.4f} |')
    lines.append(f'| Sample size (days) | {psr_info["n_days"]} |')
    lines.append('')

    lines.append('## Cost grid (V13-bear-invert)')
    lines.append('')
    lines.append('| Cost bps | Mode | Sharpe | CAGR |')
    lines.append('|---:|:--|---:|---:|')
    for cb in COST_GRID_BPS:
        for mode in ('near_close', 'one_day_lag'):
            rr = cost_grid_results[(cb, mode)]
            lines.append(f'| {cb:.1f} | {mode} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    lines.append('## Cross-variants comparison (5 bps near_close)')
    lines.append('')
    lines.append('| Variant | Sharpe | CAGR |')
    lines.append('|---|---:|---:|')
    for v in CROSS_VARIANTS:
        rr = cross_runs_at_gate_cost[v]
        lines.append(f'| {v} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')
    lines.append(
        'Documentary references (no re-run): V11 @ 5bps near_close = '
        f'{V11_REF_SHARPE_AT_5BPS_NEAR_CLOSE:.3f}, V11 @ 5bps lag = '
        f'{V11_REF_SHARPE_AT_5BPS_LAG:.3f}, V12 @ 5bps near_close = '
        f'{V12_REF_SHARPE_AT_5BPS_NEAR_CLOSE:.3f}, V12 @ 5bps lag = '
        f'{V12_REF_SHARPE_AT_5BPS_LAG:.3f}.'
    )
    lines.append('')

    lines.append('## PBO')
    lines.append('')
    lines.append(f'- Matrix shape: {matrix_shape[0]} x {matrix_shape[1]} '
                 f'({", ".join(CROSS_VARIANTS)})')
    lines.append(f'- s (CSCV submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_value:.4f}')
    lines.append(f'- Interpretation: {pbo_value:.4f} '
                 f'{"<" if pbo_pass else ">="} {PBO_THRESHOLD} = '
                 f'{"low" if pbo_pass else "high"} overfitting evidence')
    lines.append(
        '- **Methodology decision (PBO scope)**: V13 PBO is computed across '
        '7 variants (V01, V04, V05, V06, V11, V12, V13) -- V12 included so V12 '
        'vs V13 is a direct PBO neighbor (the sign-inversion test motivating V13 '
        'is most informative when V12 is in the matrix). V12 readiness ran '
        'across 6 variants (no V13). This is a deliberate expansion, documented '
        'here for reproducibility.'
    )
    lines.append('')

    lines.append('## Detector-onset alignment panel (V13 SPY-holding response)')
    lines.append('')
    events = alignment.get('events', [])
    if not events:
        lines.append('_No BEAR onsets in the test window._')
    else:
        lines.append('Per-onset breakdown of V13 SPY-holding response. `gap_days` = '
                     'trading-day gap between actual detector BEAR onset and the '
                     'SPY drawdown trough within [-20d, +30d]; positive = detector '
                     'late vs trough. `spy_return` = SPY return during the '
                     'contiguous BEAR window (V13 holds 100% SPY here).')
        lines.append('')
        lines.append('| Onset | Window | BEAR window | BEAR days | SPY trough | Gap days | SPY return |')
        lines.append('|---|---|---|---:|---|---:|---:|')
        for e in events:
            lines.append(
                f'| {e["onset"]} | {e["window_start"]} .. {e["window_end"]} | '
                f'{e["bear_start"]} .. {e["bear_end"]} | {e["bear_days"]} | '
                f'{e["spy_trough_date"]} | {e["gap_days"]:+d} | '
                f'{_fmt_pct(e["spy_return"])} |'
            )
        lines.append('')
        lines.append(f'**Aggregate gap days (mean)**: {alignment["aggregate_gap_days"]:+.2f}')
        lines.append(f'**Aggregate SPY return during BEAR window (mean)**: '
                     f'{_fmt_pct(alignment["aggregate_spy_return"])}')
        lines.append('')
        lines.append(
            'Interpretation: if mean gap_days is negative (detector late vs '
            'trough) AND mean SPY return during the BEAR window is positive, the '
            'BEAR-as-buy hypothesis is empirically supported on this sample. '
            'Sign of mean SPY return is the headline observable; magnitude '
            'feeds the gate Sharpe via daily returns.'
        )
    lines.append('')

    lines.append('## Limitations and honesty discipline')
    lines.append('')
    lines.append(
        '- **NOT OOS in strict sense.** V13 was generated from inspection of '
        "V12's 2017-2026 BEAR onset panel. The same window is now the test "
        'window. PSR/DSR partially correct for this via n_trials_project=23, '
        'but the correction is not perfect; the only definitive check is '
        'forward OOS data.'
    )
    lines.append(
        '- **Single-name concentration risk.** V13 collapses to 100% SPY on '
        'BEAR days. This is concentrated single-name risk that V11/V12 do not '
        'carry. Position sizing risk gates (production strategy framework) '
        'would need to relax for V13 deploy.'
    )
    lines.append(
        '- **Detector lag is structural, not random.** The gap_days mean is '
        'driven by SPY-DD / VIX-percentile / momentum-slope thresholds in the '
        'detector. If WS-3 improves the detector (earlier BEAR firing), V13 '
        'edge would shrink. V13 is conditional on the current detector spec.'
    )
    lines.append(
        '- **No sensitivity appendix.** V13 has no UNPREDICTABLE-cash or '
        'debouncing analog (BEAR is the only branch that differs from V11), '
        "so the V12 readiness sensitivity slice doesn't translate."
    )
    lines.append('')

    lines.append('## Methodology decisions')
    lines.append('')
    lines.append(f'- n_trials_project = {n_trials_used} ({n_trials_source})')
    lines.append(f'- PBO s = {PBO_S} (methodology Section 2.4 default)')
    lines.append(f'- PBO matrix variants: {", ".join(CROSS_VARIANTS)} (7-variant; '
                 'see PBO section for rationale).')
    lines.append(f'- Cost tier for PSR/DSR/PBO: {GATE_COST_BPS} bps per side')
    lines.append(
        '- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, '
        'trades executed at close T+1, MTM at close T+1.'
    )
    lines.append('- All metrics computed on net-of-cost daily returns.')
    lines.append('- Variants run at full window (start..end). Phase 4 is single-pass, '
                 'not walk-forward.')
    lines.append('- Gate 4 (rev4 directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- '
                 'lag > near_close is the safe direction and is not penalized.')
    lines.append('- Gate 5 (rev4-followup): both clauses required: '
                 'Sharpe(V13 @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).')
    lines.append(
        '- TIER 1 lift gate: Sharpe(V13 @ 5bps nc) > Sharpe(V11 @ 5bps nc) + 0.10. '
        'Required in addition to passing all 5 gates for TIER 1 verdict (per V13 spec).'
    )
    lines.append('')

    lines.append('## Metadata')
    lines.append('')
    lines.append(f'- Git SHA: {sha}')
    lines.append(f'- Run datetime: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'- n_trials_project source: {n_trials_source}')
    lines.append(f'- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = {v11_ref:.4f}; '
                 f'V11-readiness-doc value = {V11_REF_SHARPE_AT_7P5BPS_LAG:.4f}')
    lines.append(f'- Total gate-influencing runs: 15 '
                 '(8 V13 cost grid + 6 cross-variants + 1 V11 ref). '
                 'No sensitivity appendix runs.')
    lines.append('')

    return '\n'.join(lines)


def _smoke_run(args) -> int:
    """Run ONE V13 backtest @ 5bps near_close and exit."""
    logger.info('[+] Smoke mode: one V13-bear-invert backtest @ 5bps near_close')
    t0 = time.time()
    rr = _run(args, 'V13-bear-invert', 5.0, 'near_close', label='V13-smoke')
    elapsed = time.time() - t0
    logger.info(f'[+] Smoke complete in {elapsed:.1f}s. Sharpe={rr.sharpe:.4f}, '
                f'CAGR={rr.cagr:.2%}, n_days={len(rr.records)}')
    return 0


def main() -> int:
    args = _parse_args()

    if args.smoke:
        return _smoke_run(args)

    logger.info(
        f'[+] Starting V13 readiness orchestrator: window {args.start.date()}..{args.end.date()}'
    )
    t_start = time.time()

    # ----- Cost grid (8 runs: V13 across 4 cost tiers x 2 timing modes) -----
    cost_grid: Dict[Tuple[float, str], RunResult] = {}
    for cb in COST_GRID_BPS:
        for mode in ('near_close', 'one_day_lag'):
            cost_grid[(cb, mode)] = _run(args, 'V13-bear-invert', cb, mode)

    # ----- Cross-variants at 5 bps near_close (6 runs; V13 reused from grid) -----
    cross_runs_at_gate_cost: Dict[str, RunResult] = {}
    cross_runs_at_gate_cost['V13-bear-invert'] = cost_grid[(GATE_COST_BPS, 'near_close')]
    for v in CROSS_VARIANTS:
        if v == 'V13-bear-invert':
            continue
        cross_runs_at_gate_cost[v] = _run(args, v, GATE_COST_BPS, 'near_close')

    # ----- V11 reference for Gate 5 (1 run) -----
    v11_lag_7p5 = _run(args, 'V11', 7.5, 'one_day_lag', label='V11-ref-7p5bps-lag')

    logger.info('[+] All 15 runs complete. Computing PSR / DSR / PBO / alignment...')

    # ----- PBO across 7 variants at 5bps near_close -----
    records_by_variant = {v: cross_runs_at_gate_cost[v].records for v in CROSS_VARIANTS}
    matrix = _build_pbo_matrix(records_by_variant)
    pbo_value = pbo(matrix, s=PBO_S)
    logger.info(f'[+] PBO: {pbo_value:.4f} (matrix {matrix.shape})')

    # ----- PSR for V13 @ 5bps near_close -----
    v13_5_nc = cost_grid[(GATE_COST_BPS, 'near_close')]
    psr_info = _compute_psr(v13_5_nc.records)
    logger.info(f'[+] PSR (V13 vs SR=0): {psr_info["psr"]:.4f}')

    # ----- DSR (n_trials_project hard-coded = 23) -----
    n_trials_project = N_TRIALS_PROJECT
    n_trials_source = N_TRIALS_SOURCE
    logger.info(f'[+] n_trials_project = {n_trials_project} ({n_trials_source})')

    trial_sharpes_annual = [cross_runs_at_gate_cost[v].sharpe for v in CROSS_VARIANTS]
    trial_sharpes_daily = [s / float(np.sqrt(252)) for s in trial_sharpes_annual]
    sr_zero_daily = expected_max_sharpe(trial_sharpes_daily, n_trials_project)
    sr_zero_annual = sr_zero_daily * float(np.sqrt(252))
    dsr_value = dsr(
        sr_hat=psr_info['sr_daily'],
        trial_sharpes=trial_sharpes_daily,
        n=psr_info['n_days'],
        skew=psr_info['skew'],
        kurt=psr_info['pearson_kurt'],
        n_trials_project=n_trials_project,
    )
    v_sqrt = float(np.sqrt(np.var(trial_sharpes_daily, ddof=1)))
    dsr_info = {
        'dsr': float(dsr_value),
        'expected_max_sharpe_daily': float(sr_zero_daily),
        'expected_max_sharpe_annual': float(sr_zero_annual),
        'trial_sharpes_str': ', '.join(
            f'{v}={cross_runs_at_gate_cost[v].sharpe:.3f}' for v in CROSS_VARIANTS
        ),
        'v_sqrt_daily': v_sqrt,
    }
    logger.info(
        f'[+] DSR: prob={dsr_value:.4f} expected_max_annual={sr_zero_annual:.4f} '
        f'v_sqrt_daily={v_sqrt:.6f}'
    )

    # ----- Detector-onset alignment panel -----
    panel = load_universe_panel(args.universe, args.start, args.end)
    alignment = _detector_onset_alignment(v13_5_nc.records, panel)
    logger.info(f'[+] Alignment: {len(alignment["events"])} BEAR onsets, '
                f'aggregate gap_days={alignment["aggregate_gap_days"]:+.2f}, '
                f'aggregate spy_return={alignment["aggregate_spy_return"]:+.4f}')

    # ----- Render report -----
    sha = _git_sha()
    md = _build_doc(
        args=args,
        sha=sha,
        cost_grid_results=cost_grid,
        cross_runs_at_gate_cost=cross_runs_at_gate_cost,
        v11_lag_7p5=v11_lag_7p5,
        psr_info=psr_info,
        dsr_info=dsr_info,
        pbo_value=pbo_value,
        matrix_shape=matrix.shape,
        alignment=alignment,
        n_trials_used=n_trials_project,
        n_trials_source=n_trials_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    total_elapsed = time.time() - t_start
    logger.info(f'[+] Wrote {args.output}')
    logger.info(f'[+] Total wall-clock: {total_elapsed / 60:.2f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
