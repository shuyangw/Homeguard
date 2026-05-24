#!/usr/bin/env python3
"""V12 Phase D readiness orchestrator.

Runs the 18-backtest battery required to gate V12 (BEAR-to-cash on V11 base)
for paper deployment:

  GATE-INFLUENCING (14 runs):
    - V12 cost grid: {1, 5, 7.5, 10} bps x {near_close, one_day_lag} = 8 runs
    - Cross-variants at 5 bps near_close: V01, V04, V05, V06, V11 = 5 runs
      (V12 @ 5 bps near_close already in the cost grid; no double count)
    - V11 @ 7.5 bps one_day_lag (Gate 5 reference) = 1 run

  SENSITIVITY APPENDIX (4 runs -- NOT gate-influencing; do increment
  n_trials_project for DSR conservatism):
    - V12-up-cash: UNPREDICTABLE='cash' @ 5 bps near_close = 1 run
    - V12-deb-2/3/5: min_regime_days = {2, 3, 5} @ 5 bps near_close = 3 runs

Five gates (rev4 + rev4-followup):
  1. PSR(V12 @ 5bps near_close)            > 0.95          (vs SR=0)
  2. DSR(V12)                              > 0.95          (project-wide n_trials)
  3. PBO across 6 variants                 < 0.5           (CSCV s=16)
  4. lag delta                             <= max(0.2*nc, 0.1)
  5a. Sharpe(V12 @ 7.5bps one_day_lag)     > 0.30
  5b. Sharpe(V12 @ 7.5bps one_day_lag)     >= 0.9 * Sharpe(V11 @ 7.5bps one_day_lag)

Plus a detector-onset alignment panel: for each BEAR onset in V12's record
stream, report the cash window and the gap-days between the actual onset
and the SPY drawdown trough within the +/-20-day window (a practical
lag-tax proxy).
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
}

# 6 variants used for PBO matrix at 5 bps near_close.
CROSS_VARIANTS: Tuple[str, ...] = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')
GATE_TARGET = 'V12'

# Spec rev4: (1, 5, 7.5, 10) bps. Different from V11 readiness's (0, 2.5, 5, 7.5).
COST_GRID_BPS: Tuple[float, ...] = (1.0, 5.0, 7.5, 10.0)
GATE_COST_BPS = 5.0

# n_trials_project: resolved at runtime from experiments.duckdb when possible.
N_TRIALS_FALLBACK = 30
EXPERIMENTS_DB_PATH = Path('output/experiments.duckdb')

PBO_S = 16

# Gate thresholds.
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
LAG_DEGRADATION_FRACTION = 0.2   # relative cap (rev4)
LAG_DEGRADATION_FLOOR = 0.1      # absolute cap (rev4 floor)
COST_FLOOR_SHARPE = 0.3          # rev4-followup
COST_NO_REGRESS_FRACTION = 0.9   # V12 >= 0.9 * V11 at 7.5bps one_day_lag

# V11 reference Sharpe pulled from
# docs/reports/ramp/20260523_phase4_v11_readiness.md (table at line ~74).
# Documentary fallback only; we ALSO re-run V11 @ 7.5bps one_day_lag inline.
V11_REF_SHARPE_AT_7P5BPS_LAG = 0.5306


@dataclass
class RunResult:
    variant_id: str
    cost_bps: float
    timing_mode: str
    records: List[DailyRecord]
    sharpe: float
    cagr: float
    label: str = ''  # optional human-readable tag (e.g. 'V12-up-cash')


def _git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='V12 Phase D readiness orchestrator.')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke mode: run ONE V12 backtest @ 5bps near_close and exit (no report).')
    p.add_argument('--start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2017, 1, 1))
    p.add_argument('--end', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2026, 5, 16))
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path,
                   default=Path('docs/reports/ramp/20260524_phase4_v12_readiness.md'))
    return p.parse_args(argv)


def _resolve_n_trials_project(extra_in_this_run: int) -> Tuple[int, str]:
    """Read project-wide cumulative trial count from experiments.duckdb.

    Per methodology Section 9.4 / strategy-pipeline rules, n_trials_project
    is the cumulative count across the project. We add `extra_in_this_run`
    (the runs about to be registered by this orchestrator) so the DSR
    correction reflects the run we're computing it for.

    Falls back to N_TRIALS_FALLBACK if the registry is absent or empty.
    """
    if not EXPERIMENTS_DB_PATH.exists():
        return N_TRIALS_FALLBACK, f'fallback (no {EXPERIMENTS_DB_PATH})'
    try:
        import duckdb
        con = duckdb.connect(str(EXPERIMENTS_DB_PATH), read_only=True)
        total = con.execute('SELECT COALESCE(SUM(combinations_in_run), 0) FROM runs').fetchone()[0]
        con.close()
        total = int(total or 0)
        if total <= 0:
            return N_TRIALS_FALLBACK + extra_in_this_run, (
                f'fallback (registry empty: total={total})'
            )
        return total + extra_in_this_run, f'experiments.duckdb total={total} + {extra_in_this_run}'
    except Exception as exc:
        return N_TRIALS_FALLBACK, f'fallback (registry read failed: {exc})'


def _build_cfg(args, variant_id: str, cost_bps: float, timing_mode: str,
               regime_positions: Optional[Dict[str, str]] = None,
               min_regime_days: int = 0) -> HarnessConfig:
    delta_pct = DELTA_REBALANCE_PCT_BY_VARIANT.get(variant_id, 0.0)
    kwargs = dict(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=delta_pct,
    )
    if regime_positions is not None:
        kwargs['regime_positions'] = regime_positions
    if min_regime_days:
        kwargs['min_regime_days'] = min_regime_days
    return HarnessConfig(**kwargs)


def _run(args, variant_id: str, cost_bps: float, timing_mode: str,
         label: str = '',
         regime_positions: Optional[Dict[str, str]] = None,
         min_regime_days: int = 0) -> RunResult:
    cfg = _build_cfg(args, variant_id, cost_bps, timing_mode,
                     regime_positions=regime_positions,
                     min_regime_days=min_regime_days)
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
    """PSR(SR_hat | SR_benchmark=0) in per-period (daily) units (methodology 2.2).

    Mertens (2002) asymptotic variance applies to the per-period estimator;
    passing annualized SR inflates z by sqrt(P).
    """
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
    """For each BEAR onset, report cash window + lag-tax proxy.

    Lag-tax proxy: trading-day gap between actual detector BEAR onset and the
    SPY drawdown trough within [onset - 20d, onset + 30d]. Positive gap means
    the detector fired AFTER the trough (late). Negative means BEFORE.
    The detector-perfect counterfactual replay is out of scope; the gap is
    the practical proxy the reader interprets.
    """
    if not records or 'SPY' not in panel.columns:
        return {'events': [], 'aggregate_gap_days': 0.0, 'aggregate_avoided_return': 0.0}

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

        # Cash window: contiguous days from onset where realized_weights is empty.
        cash_start = onset
        cash_end = onset
        for j in range(idx_onset, len(records)):
            if len(records[j].realized_weights) == 0:
                cash_end = pd.Timestamp(records[j].date)
            else:
                break

        # SPY trajectory window for the panel display.
        win_lo = onset - pd.Timedelta(days=20)
        win_hi = onset + pd.Timedelta(days=30)
        spy_window = spy.loc[(spy.index >= win_lo) & (spy.index <= win_hi)]

        # SPY return during cash window: the return V12 "avoided" by being out.
        spy_cash_window = spy.loc[(spy.index >= cash_start) & (spy.index <= cash_end)]
        if len(spy_cash_window) >= 2:
            avoided_return = float(spy_cash_window.iloc[-1] / spy_cash_window.iloc[0] - 1.0)
        else:
            avoided_return = 0.0

        # Gap days = trading-day delta between SPY drawdown trough (within window)
        # and the actual BEAR onset. Positive = detector late.
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
            'cash_start': str(cash_start.date()),
            'cash_end': str(cash_end.date()),
            'cash_days': int((cash_end - cash_start).days) + 1,
            'spy_trough_date': str(trough_date.date()) if trough_date is not None else None,
            'gap_days': int(gap_days),
            'avoided_return': float(avoided_return),
        })

    n = len(events)
    return {
        'events': events,
        'aggregate_gap_days': float(sum(e['gap_days'] for e in events) / n) if n else 0.0,
        'aggregate_avoided_return': float(sum(e['avoided_return'] for e in events) / n) if n else 0.0,
    }


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def _build_doc(
    args,
    sha: str,
    cost_grid_results: Dict[Tuple[float, str], RunResult],
    cross_runs_at_gate_cost: Dict[str, RunResult],
    v11_lag_7p5: RunResult,
    sensitivity_results: List[RunResult],
    psr_info: Dict[str, float],
    dsr_info: Dict[str, float],
    pbo_value: float,
    matrix_shape: tuple,
    alignment: Dict[str, object],
    n_trials_used: int,
    n_trials_source: str,
) -> str:
    nc_v12_5 = cost_grid_results[(GATE_COST_BPS, 'near_close')].sharpe
    lag_v12_5 = cost_grid_results[(GATE_COST_BPS, 'one_day_lag')].sharpe
    v12_lag_7p5 = cost_grid_results[(7.5, 'one_day_lag')].sharpe

    # Gate 1: PSR
    psr_value = psr_info['psr']
    psr_pass = psr_value > PSR_THRESHOLD

    # Gate 2: DSR
    dsr_value = dsr_info['dsr']
    dsr_pass = dsr_value > DSR_THRESHOLD

    # Gate 3: PBO
    pbo_pass = pbo_value < PBO_THRESHOLD

    # Gate 4: lag delta with rev4 floor
    delta_abs = nc_v12_5 - lag_v12_5
    cap = max(LAG_DEGRADATION_FRACTION * abs(nc_v12_5), LAG_DEGRADATION_FLOOR)
    lag_pass = abs(delta_abs) <= cap

    # Gate 5: cost-floor + no-regress (both clauses)
    v11_ref = v11_lag_7p5.sharpe
    cost_floor_pass = v12_lag_7p5 > COST_FLOOR_SHARPE
    no_regress_pass = v12_lag_7p5 >= COST_NO_REGRESS_FRACTION * v11_ref
    cost_pass = cost_floor_pass and no_regress_pass

    all_gates = [psr_pass, dsr_pass, pbo_pass, lag_pass, cost_pass]
    if all(all_gates):
        verdict = 'READY for Phase D paper deploy'
        verdict_short = 'PASS'
    elif (pbo_pass and lag_pass and cost_pass) and not (psr_pass and dsr_pass):
        verdict = (
            'PARTIAL -- passes structural + cost gates (PBO, lag, cost-no-regress); '
            'fails absolute-significance gates (PSR/DSR). Decision to advance is a '
            'judgment call.'
        )
        verdict_short = 'PARTIAL'
    else:
        verdict = 'BLOCKED -- one or more structural/cost gates failed; investigate before Phase D'
        verdict_short = 'FAIL'

    def _pf(b: bool) -> str:
        return 'PASS' if b else 'FAIL'

    lines: List[str] = []
    lines.append('# V12 Phase D Readiness Report')
    lines.append('')
    lines.append(f'**Code commit**: {sha}')
    lines.append(f'**Data**: Alpaca SIP daily-aggregated, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Cost tier for PSR/DSR/PBO gates**: {GATE_COST_BPS} bps per side')
    lines.append(f'**n_trials_project**: {n_trials_used} ({n_trials_source})')
    lines.append('')

    lines.append('## Summary')
    lines.append('')
    lines.append('| Gate | Result | Value | Threshold |')
    lines.append('|---|:---:|---:|---:|')
    lines.append(f'| 1. PSR(V12 @ 5bps near_close, vs SR=0) | {_pf(psr_pass)} | {psr_value:.4f} | > {PSR_THRESHOLD} |')
    lines.append(f'| 2. DSR(V12, n_trials={n_trials_used}) | {_pf(dsr_pass)} | {dsr_value:.4f} | > {DSR_THRESHOLD} |')
    lines.append(
        f'| 3. PBO across {{V01,V04,V05,V06,V11,V12}} | {_pf(pbo_pass)} | '
        f'{pbo_value:.4f} | < {PBO_THRESHOLD} |'
    )
    lines.append(
        f'| 4. Lag-degradation (5 bps) | {_pf(lag_pass)} | '
        f'nc={nc_v12_5:.3f}, lag={lag_v12_5:.3f}, delta={delta_abs:+.3f} | '
        f'<= max(0.2*nc, 0.1) = {cap:.3f} |'
    )
    lines.append(
        f'| 5a. Sharpe(V12 @ 7.5bps lag) > 0.30 | {_pf(cost_floor_pass)} | '
        f'{v12_lag_7p5:.4f} | > {COST_FLOOR_SHARPE} |'
    )
    lines.append(
        f'| 5b. Sharpe(V12) >= 0.9 * Sharpe(V11) @ 7.5bps lag | {_pf(no_regress_pass)} | '
        f'V12={v12_lag_7p5:.4f}, 0.9*V11={COST_NO_REGRESS_FRACTION * v11_ref:.4f} '
        f'(V11={v11_ref:.4f}) | >= {COST_NO_REGRESS_FRACTION} * V11 |'
    )
    lines.append('')
    lines.append(f'**Overall**: {verdict_short} -- {verdict}')
    lines.append('')

    lines.append('## Headline 5-gate verdict')
    lines.append('')
    lines.append('See Summary table above. Gates 1-3 are the structural / significance trio; '
                 'gate 4 is the rev4 lag-floor robustness check; gate 5 is the rev4-followup '
                 'cost-floor + no-regress dual clause.')
    lines.append('')
    lines.append('Gate 5 V11 reference (Sharpe @ 7.5bps one_day_lag) was re-run inline in '
                 'this orchestrator for an apples-to-apples comparison. '
                 f'Inline V11 value: {v11_ref:.4f}. For reference, the V11 readiness doc '
                 f'(2026-05-23) reported {V11_REF_SHARPE_AT_7P5BPS_LAG:.4f}.')
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
    lines.append(f'| Observed Sharpe (V12) | {psr_info["sr_daily"]:.6f} | {psr_info["sr_annual"]:.4f} |')
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

    lines.append('## Cost grid (V12 v12.0.0 defaults)')
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

    lines.append('## PBO')
    lines.append('')
    lines.append(f'- Matrix shape: {matrix_shape[0]} x {matrix_shape[1]} '
                 f'({", ".join(CROSS_VARIANTS)})')
    lines.append(f'- s (CSCV submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_value:.4f}')
    lines.append(f'- Interpretation: {pbo_value:.4f} < {PBO_THRESHOLD} = low overfitting evidence')
    lines.append('')

    lines.append('## Detector-onset alignment panel')
    lines.append('')
    events = alignment.get('events', [])
    if not events:
        lines.append('_No BEAR onsets in the test window._')
    else:
        lines.append('Per-onset breakdown of V12 cash response. `gap_days` = trading-day '
                     'gap between actual detector BEAR onset and the SPY drawdown trough '
                     'within [-20d, +30d]; positive = detector late vs trough. '
                     '`avoided_return` = SPY return during the cash window (the return '
                     'V12 sidestepped by being out).')
        lines.append('')
        lines.append('| Onset | Window | Cash window | Cash days | SPY trough | Gap days | Avoided return |')
        lines.append('|---|---|---|---:|---|---:|---:|')
        for e in events:
            lines.append(
                f'| {e["onset"]} | {e["window_start"]} .. {e["window_end"]} | '
                f'{e["cash_start"]} .. {e["cash_end"]} | {e["cash_days"]} | '
                f'{e["spy_trough_date"]} | {e["gap_days"]:+d} | '
                f'{_fmt_pct(e["avoided_return"])} |'
            )
        lines.append('')
        lines.append(f'**Aggregate gap days (mean)**: {alignment["aggregate_gap_days"]:+.2f}')
        lines.append(f'**Aggregate avoided return (mean)**: '
                     f'{_fmt_pct(alignment["aggregate_avoided_return"])}')
    lines.append('')

    lines.append('## Sensitivity appendix (NOT gate-influencing)')
    lines.append('')
    lines.append('_These runs ARE counted in `n_trials_project` (DSR conservatism) but do '
                 'NOT feed PSR/DSR/PBO/lag/cost computations._')
    lines.append('')
    lines.append('| Run | Description | Sharpe | CAGR |')
    lines.append('|---|---|---:|---:|')
    for rr in sensitivity_results:
        lines.append(f'| {rr.label} | {rr.variant_id} @ 5bps near_close (variant) | '
                     f'{rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    lines.append('## Methodology decisions')
    lines.append('')
    lines.append(f'- n_trials_project = {n_trials_used} ({n_trials_source}; '
                 'methodology Section 2.3 / 9.4 cumulative)')
    lines.append(f'- PBO s = {PBO_S} (methodology Section 2.4 default)')
    lines.append(f'- Cost tier for PSR/DSR/PBO: {GATE_COST_BPS} bps per side')
    lines.append(
        '- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, '
        'trades executed at close T+1, MTM at close T+1.'
    )
    lines.append('- All metrics computed on net-of-cost daily returns.')
    lines.append('- Variants run at full window (start..end). Phase 4 is single-pass, '
                 'not walk-forward.')
    lines.append('- Gate 4 floor (rev4): `|nc - lag| <= max(0.2 * |nc|, 0.1)` -- absolute '
                 'floor of 0.1 prevents trivially-passing tiny-Sharpe runs.')
    lines.append('- Gate 5 (rev4-followup): both clauses required: '
                 'Sharpe(V12 @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).')
    lines.append('')

    lines.append('## Metadata')
    lines.append('')
    lines.append(f'- Git SHA: {sha}')
    lines.append(f'- Run datetime: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'- n_trials_project source: {n_trials_source}')
    lines.append(f'- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = {v11_ref:.4f}; '
                 f'V11-readiness-doc value = {V11_REF_SHARPE_AT_7P5BPS_LAG:.4f}')
    lines.append(f'- Total gate-influencing runs: 14 (8 cost grid + 5 cross + 1 V11 ref)')
    lines.append(f'- Total sensitivity runs: {len(sensitivity_results)}')
    lines.append('')

    return '\n'.join(lines)


def _smoke_run(args) -> int:
    """Run ONE V12 backtest @ 5bps near_close and exit."""
    logger.info('[+] Smoke mode: one V12 backtest @ 5bps near_close')
    t0 = time.time()
    rr = _run(args, 'V12', 5.0, 'near_close', label='V12-smoke')
    elapsed = time.time() - t0
    logger.info(f'[+] Smoke complete in {elapsed:.1f}s. Sharpe={rr.sharpe:.4f}, '
                f'CAGR={rr.cagr:.2%}, n_days={len(rr.records)}')
    return 0


def main() -> int:
    args = _parse_args()

    if args.smoke:
        return _smoke_run(args)

    logger.info(
        f'[+] Starting V12 readiness orchestrator: window {args.start.date()}..{args.end.date()}'
    )
    t_start = time.time()

    # ----- Cost grid (8 runs: V12 across 4 cost tiers x 2 timing modes) -----
    cost_grid: Dict[Tuple[float, str], RunResult] = {}
    for cb in COST_GRID_BPS:
        for mode in ('near_close', 'one_day_lag'):
            cost_grid[(cb, mode)] = _run(args, 'V12', cb, mode)

    # ----- Cross-variants at 5 bps near_close (5 runs; V12 reused from grid) -----
    cross_runs_at_gate_cost: Dict[str, RunResult] = {}
    cross_runs_at_gate_cost['V12'] = cost_grid[(GATE_COST_BPS, 'near_close')]
    for v in CROSS_VARIANTS:
        if v == 'V12':
            continue
        cross_runs_at_gate_cost[v] = _run(args, v, GATE_COST_BPS, 'near_close')

    # ----- V11 reference for Gate 5 (1 run) -----
    v11_lag_7p5 = _run(args, 'V11', 7.5, 'one_day_lag', label='V11-ref-7p5bps-lag')

    # ----- Sensitivity appendix (4 runs; informational) -----
    sensitivity_results: List[RunResult] = []

    # V12-up-cash: UNPREDICTABLE='cash'.
    up_cash_positions = {
        'STRONG_BULL':   'normal',
        'WEAK_BULL':     'normal',
        'SIDEWAYS':      'normal',
        'UNPREDICTABLE': 'cash',
        'BEAR':          'cash',
        'SAFE_MODE':     'hold',
    }
    sensitivity_results.append(_run(
        args, 'V12', GATE_COST_BPS, 'near_close',
        label='V12-up-cash',
        regime_positions=up_cash_positions,
    ))

    # V12-deb-{2,3,5}: min_regime_days = 2 / 3 / 5.
    for d in (2, 3, 5):
        sensitivity_results.append(_run(
            args, 'V12', GATE_COST_BPS, 'near_close',
            label=f'V12-deb-{d}',
            min_regime_days=d,
        ))

    logger.info('[+] All 18 runs complete. Computing PSR / DSR / PBO / alignment...')

    # ----- PBO across 6 variants at 5bps near_close -----
    records_by_variant = {v: cross_runs_at_gate_cost[v].records for v in CROSS_VARIANTS}
    matrix = _build_pbo_matrix(records_by_variant)
    pbo_value = pbo(matrix, s=PBO_S)
    logger.info(f'[+] PBO: {pbo_value:.4f} (matrix {matrix.shape})')

    # ----- PSR for V12 @ 5bps near_close -----
    v12_5_nc = cost_grid[(GATE_COST_BPS, 'near_close')]
    psr_info = _compute_psr(v12_5_nc.records)
    logger.info(f'[+] PSR (V12 vs SR=0): {psr_info["psr"]:.4f}')

    # ----- n_trials_project: read from experiments.duckdb (+ 18 from this run) -----
    n_trials_project, n_trials_source = _resolve_n_trials_project(extra_in_this_run=18)
    logger.info(f'[+] n_trials_project = {n_trials_project} ({n_trials_source})')

    # ----- DSR -----
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
    alignment = _detector_onset_alignment(v12_5_nc.records, panel)
    logger.info(f'[+] Alignment: {len(alignment["events"])} BEAR onsets, '
                f'aggregate gap_days={alignment["aggregate_gap_days"]:+.2f}')

    # ----- Render report -----
    sha = _git_sha()
    md = _build_doc(
        args=args,
        sha=sha,
        cost_grid_results=cost_grid,
        cross_runs_at_gate_cost=cross_runs_at_gate_cost,
        v11_lag_7p5=v11_lag_7p5,
        sensitivity_results=sensitivity_results,
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
