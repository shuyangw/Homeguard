#!/usr/bin/env python3
"""V12c Phase D readiness orchestrator (Experiment 6).

V12c = V12 plan_fn (BEAR-to-cash) + UNPREDICTABLE also to cash. Same code as
V12; differs only in `cfg.regime_positions[UNPREDICTABLE]='cash'`. V12c was
discovered as the V12-up-cash sensitivity finding (Sharpe 0.586 vs V12 default
0.268 in the 2026-05-24 V12 readiness report) and is now being put through its
own formal 5-gate readiness gate per spec rev4 honesty discipline.

Runs 15 unique backtests (17 logical, counting one reuse + one post-hoc
metric):

  GATE-INFLUENCING (15 runs):
    - V12c cost grid: {1, 5, 7.5, 10} bps x {near_close, one_day_lag} = 8 runs
    - Cross-variants at 5 bps near_close:
        V01, V04, V05, V06, V11, V12 (default BEAR-only cash), V12c (this one)
        = 7 variants; V12c at 5 bps near_close already in the cost grid, so 6
        NEW runs here.
    - V11 reference at 7.5 bps one_day_lag (Gate 5 no-regress baseline) = 1 run

  SENSITIVITY APPENDIX (INFORMATIONAL, NOT gate-influencing):
    - COVID-excluded subgroup: re-compute V12c Sharpe at 5 bps near_close
      after dropping trading days in 2020-02-24 .. 2020-04-30 (inclusive).
      Implemented as a post-hoc filter of the cost-grid V12c@5bps-near_close
      record stream (NOT a fresh backtest, so no PSR/DSR distortion).

Five gates (rev4 + rev4-followup):
  1. PSR(V12c @ 5bps near_close)             > 0.95          (vs SR=0)
  2. DSR(V12c)                                > 0.95          (n_trials=23)
  3. PBO across 7 variants                    < 0.5           (CSCV s=16)
  4. lag delta                                <= max(0.2*nc, 0.1)
  5a. Sharpe(V12c @ 7.5bps one_day_lag)       > 0.30
  5b. Sharpe(V12c @ 7.5bps one_day_lag)       >= 0.9 * Sharpe(V11 @ 7.5bps lag)

E2 (UNPREDICTABLE hand-inspection) verdict was AMBIGUOUS (top-3 attribution
share 53.6%, COVID-event-dominant). Per the conditional-proceed protocol from
the analyst decision, this orchestrator INCLUDES the COVID-excluded subgroup
panel; the gate verdict stands on the full-window numbers per spec rev4
honesty discipline. The report flags COVID concentration prominently.

E4 (V12 lag-asymmetry decomposition) verdict was DIFFUSE (38.1% transition-day
share, below 50%). The prescribed "add 10 bps stress" does NOT apply -- we use
the standard cost grid {1, 5, 7.5, 10} bps.
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
    # V12c uses V12 code with regime_positions override; same delta as V11/V12.
    'V12c': 0.02,
}

# 7 variants used for PBO matrix at 5 bps near_close. V12c added vs V12 readiness.
# The label 'V12c' is logical-only; the underlying variant in REGISTRY is 'V12'
# with an UNPREDICTABLE='cash' regime_positions override applied at cfg build time.
CROSS_LABELS: Tuple[str, ...] = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12', 'V12c')
GATE_TARGET_LABEL = 'V12c'

# V12c regime_positions: V11 base + BEAR -> cash + UNPREDICTABLE -> cash.
# This is the discovery configuration from the V12 readiness sensitivity grid
# (V12-up-cash); formalized as V12c here.
V12C_REGIME_POSITIONS: Dict[str, str] = {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'cash',
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
}

# Spec rev4: (1, 5, 7.5, 10) bps. E4 verdict DIFFUSE -> standard cost grid (no 10bps stress add).
COST_GRID_BPS: Tuple[float, ...] = (1.0, 5.0, 7.5, 10.0)
GATE_COST_BPS = 5.0

# n_trials_project is HARD-CODED at 23 here.
# - V12 readiness orchestrator used n_trials_project=22 (4 from experiments.duckdb at the
#   time + 18 runs registered by the V12 orchestrator).
# - V12c was previously counted as a SENSITIVITY run within V12's 22 (V12-up-cash, which
#   is exactly V12c's configuration). Formalizing it as its own readiness gate increments
#   the project-wide trial count by one (the formal evaluation is a distinct trial from
#   the sensitivity discovery), giving 23.
# We do NOT add the runs in this orchestrator to the count: their cross-variant Sharpes
# feed the DSR trial set explicitly, and counting them again would double-count.
N_TRIALS_PROJECT = 23
EXPERIMENTS_DB_PATH = Path('output/experiments.duckdb')

PBO_S = 16

# Gate thresholds.
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
LAG_DEGRADATION_FRACTION = 0.2   # relative cap (rev4)
LAG_DEGRADATION_FLOOR = 0.1      # absolute cap (rev4 floor)
COST_FLOOR_SHARPE = 0.3          # rev4-followup
COST_NO_REGRESS_FRACTION = 0.9   # V12c >= 0.9 * V11 at 7.5bps one_day_lag

# V11 reference Sharpe pulled from
# docs/reports/ramp/20260523_phase4_v11_readiness.md.
V11_REF_SHARPE_AT_7P5BPS_LAG = 0.5306

# COVID exclusion window (inclusive). E2-required robustness check.
COVID_EXCLUSION_START = pd.Timestamp('2020-02-24')
COVID_EXCLUSION_END = pd.Timestamp('2020-04-30')


@dataclass
class RunResult:
    label: str                # logical label, e.g. 'V12c'
    variant_id: str           # underlying REGISTRY id, e.g. 'V12'
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
    p = argparse.ArgumentParser(description='V12c Phase D readiness orchestrator.')
    p.add_argument('--smoke', action='store_true',
                   help='Smoke mode: run ONE V12c backtest @ 5bps near_close and exit (no report).')
    p.add_argument('--start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2017, 1, 1))
    p.add_argument('--end', type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                   default=datetime(2026, 5, 16))
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path,
                   default=Path('docs/reports/ramp/20260526_phase4_v12c_readiness.md'))
    return p.parse_args(argv)


def _registry_variant_id(label: str) -> str:
    """Map a logical label (e.g. 'V12c') to the underlying REGISTRY variant id."""
    if label == 'V12c':
        return 'V12'  # V12c uses V12 code + regime_positions override
    return label


def _regime_positions_for(label: str) -> Optional[Dict[str, str]]:
    if label == 'V12c':
        return dict(V12C_REGIME_POSITIONS)
    return None


def _build_cfg(args, label: str, cost_bps: float, timing_mode: str) -> HarnessConfig:
    variant_id = _registry_variant_id(label)
    delta_pct = DELTA_REBALANCE_PCT_BY_VARIANT.get(label, 0.0)
    kwargs = dict(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=delta_pct,
    )
    rps = _regime_positions_for(label)
    if rps is not None:
        kwargs['regime_positions'] = rps
    _ = variant_id  # used inside _run below
    return HarnessConfig(**kwargs)


def _run(args, label: str, cost_bps: float, timing_mode: str) -> RunResult:
    variant_id = _registry_variant_id(label)
    cfg = _build_cfg(args, label, cost_bps, timing_mode)
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


def _build_pbo_matrix(records_by_label: Dict[str, List[DailyRecord]]) -> np.ndarray:
    anchor = CROSS_LABELS[0]
    dates_anchor = [r.date for r in records_by_label[anchor]]
    for v in CROSS_LABELS[1:]:
        dates_v = [r.date for r in records_by_label[v]]
        if dates_v != dates_anchor:
            raise RuntimeError(
                f'Date mismatch: {anchor} has {len(dates_anchor)} dates, '
                f'{v} has {len(dates_v)}'
            )
    return np.column_stack([
        np.array([r.daily_return for r in records_by_label[v]], dtype=np.float64)
        for v in CROSS_LABELS
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


def _covid_excluded_sharpe(records: List[DailyRecord]) -> Dict[str, float]:
    """Compute V12c Sharpe with COVID trading days dropped (E2 robustness check).

    Filters the daily-return stream to exclude dates in
    [COVID_EXCLUSION_START, COVID_EXCLUSION_END] inclusive, then recomputes
    annualized Sharpe over the surviving days. NOT gate-influencing; the gate
    verdict stands on the full-window numbers per spec rev4 honesty discipline.
    """
    full_rets = pd.Series(
        [r.daily_return for r in records],
        index=pd.DatetimeIndex([pd.Timestamp(r.date) for r in records]),
    )
    mask = (full_rets.index >= COVID_EXCLUSION_START) & (full_rets.index <= COVID_EXCLUSION_END)
    n_dropped = int(mask.sum())
    ex_rets = full_rets.loc[~mask]
    sr_full = sharpe_ratio(full_rets)
    sr_ex = sharpe_ratio(ex_rets)
    cg_full = (1 + full_rets).prod() ** (252 / max(len(full_rets), 1)) - 1
    cg_ex = (1 + ex_rets).prod() ** (252 / max(len(ex_rets), 1)) - 1
    return {
        'sharpe_full_window': float(sr_full),
        'sharpe_ex_covid': float(sr_ex),
        'sharpe_delta': float(sr_ex - sr_full),
        'cagr_full_window': float(cg_full),
        'cagr_ex_covid': float(cg_ex),
        'n_days_full': int(len(full_rets)),
        'n_days_dropped': n_dropped,
        'n_days_ex_covid': int(len(ex_rets)),
        'window_start': str(COVID_EXCLUSION_START.date()),
        'window_end': str(COVID_EXCLUSION_END.date()),
    }


def _detector_onset_alignment(records: List[DailyRecord],
                              panel: pd.DataFrame,
                              onset_regimes: Tuple[str, ...] = ('BEAR', 'UNPREDICTABLE')) -> Dict[str, object]:
    """For each onset of a cash regime, report cash window + lag-tax proxy.

    V12c goes to cash on BEAR AND UNPREDICTABLE, so both are onset triggers.
    """
    if not records or 'SPY' not in panel.columns:
        return {'events': [], 'aggregate_gap_days': 0.0, 'aggregate_avoided_return': 0.0}

    onset_dates: List[Tuple[pd.Timestamp, str]] = []
    for i in range(1, len(records)):
        cur = records[i].regime
        prev = records[i - 1].regime
        if cur in onset_regimes and prev not in onset_regimes:
            onset_dates.append((pd.Timestamp(records[i].date), cur))

    spy = panel['SPY'].dropna()
    record_dates = [pd.Timestamp(r.date) for r in records]

    events: List[Dict[str, object]] = []
    for onset, onset_regime in onset_dates:
        try:
            idx_onset = record_dates.index(onset)
        except ValueError:
            continue

        cash_start = onset
        cash_end = onset
        for j in range(idx_onset, len(records)):
            if len(records[j].realized_weights) == 0:
                cash_end = pd.Timestamp(records[j].date)
            else:
                break

        win_lo = onset - pd.Timedelta(days=20)
        win_hi = onset + pd.Timedelta(days=30)
        spy_window = spy.loc[(spy.index >= win_lo) & (spy.index <= win_hi)]

        spy_cash_window = spy.loc[(spy.index >= cash_start) & (spy.index <= cash_end)]
        if len(spy_cash_window) >= 2:
            avoided_return = float(spy_cash_window.iloc[-1] / spy_cash_window.iloc[0] - 1.0)
        else:
            avoided_return = 0.0

        gap_days = 0
        trough_date: Optional[pd.Timestamp] = None
        if len(spy_window) > 0:
            trough_date = spy_window.idxmin()
            mask = (spy.index >= min(onset, trough_date)) & (spy.index <= max(onset, trough_date))
            n_between = int(mask.sum())
            gap_days = (n_between - 1) if onset >= trough_date else -(n_between - 1)

        events.append({
            'onset': str(onset.date()),
            'onset_regime': onset_regime,
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
    covid_panel: Dict[str, float],
    psr_info: Dict[str, float],
    dsr_info: Dict[str, float],
    pbo_value: float,
    matrix_shape: tuple,
    alignment: Dict[str, object],
    n_trials_used: int,
    n_trials_source: str,
) -> str:
    nc_v12c_5 = cost_grid_results[(GATE_COST_BPS, 'near_close')].sharpe
    lag_v12c_5 = cost_grid_results[(GATE_COST_BPS, 'one_day_lag')].sharpe
    v12c_lag_7p5 = cost_grid_results[(7.5, 'one_day_lag')].sharpe

    # Gate 1: PSR
    psr_value = psr_info['psr']
    psr_pass = psr_value > PSR_THRESHOLD

    # Gate 2: DSR
    dsr_value = dsr_info['dsr']
    dsr_pass = dsr_value > DSR_THRESHOLD

    # Gate 3: PBO
    pbo_pass = pbo_value < PBO_THRESHOLD

    # Gate 4: lag degradation, directional per spec rev4.
    nc_minus_lag = nc_v12c_5 - lag_v12c_5
    cap = max(LAG_DEGRADATION_FRACTION * abs(nc_v12c_5), LAG_DEGRADATION_FLOOR)
    lag_pass = nc_minus_lag <= cap

    # Gate 5: cost-floor + no-regress.
    v11_ref = v11_lag_7p5.sharpe
    cost_floor_pass = v12c_lag_7p5 > COST_FLOOR_SHARPE
    no_regress_pass = v12c_lag_7p5 >= COST_NO_REGRESS_FRACTION * v11_ref
    cost_pass = cost_floor_pass and no_regress_pass

    all_structural = [pbo_pass, lag_pass, cost_pass]
    all_significance = [psr_pass, dsr_pass]
    all_gates = all_significance + all_structural

    # Tier classification per spec rev4 5-gate standard:
    # - TIER 1: all 5 gates PASS
    # - TIER 3: structural gates PASS, absolute-significance gates (PSR, DSR) FAIL
    # - TIER 4: any structural gate FAILS
    if all(all_gates):
        verdict = 'READY for Phase D paper deploy (all 5 gates PASS)'
        verdict_short = 'TIER 1'
    elif all(all_structural) and not all(all_significance):
        verdict = (
            'passes structural + cost gates (PBO, lag, cost-no-regress); '
            'fails absolute-significance gates (PSR/DSR). Decision to advance is a '
            'judgment call.'
        )
        verdict_short = 'TIER 3'
    else:
        verdict = 'one or more structural/cost gates failed; do not advance'
        verdict_short = 'TIER 4'

    def _pf(b: bool) -> str:
        return 'PASS' if b else 'FAIL'

    lines: List[str] = []
    lines.append('# V12c Phase D Readiness Report (Experiment 6)')
    lines.append('')
    lines.append(f'**Code commit**: {sha}')
    lines.append(f'**Data**: Alpaca SIP daily-aggregated, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Cost tier for PSR/DSR/PBO gates**: {GATE_COST_BPS} bps per side')
    lines.append(f'**n_trials_project**: {n_trials_used} ({n_trials_source})')
    lines.append('')
    lines.append(
        '**Variant definition**: V12c = V12 plan_fn (BEAR-to-cash) with '
        '`regime_positions[UNPREDICTABLE] = "cash"`. Discovered as the V12-up-cash '
        'sensitivity finding in the 2026-05-24 V12 readiness report; formalized here.'
    )
    lines.append('')
    lines.append(
        '**Pre-gate conditional-proceed context (E2 / E4)**: E2 hand-inspection of '
        "UNPREDICTABLE's drawdown-avoidance attribution returned AMBIGUOUS (top-3 "
        'event share 53.6%, COVID-dominant). E4 lag-asymmetry decomposition returned '
        "DIFFUSE (transition-day share 38.1%, below the 50% threshold), so the standard "
        'cost grid is used (no 10 bps stress add). Per analyst direction, this report '
        'INCLUDES a COVID-excluded subgroup panel (informational only; gates stand on '
        'full-window numbers).'
    )
    lines.append('')

    lines.append('## Summary -- 5-gate verdict')
    lines.append('')
    lines.append('| Gate | Result | Value | Threshold |')
    lines.append('|---|:---:|---:|---:|')
    lines.append(f'| 1. PSR(V12c @ 5bps near_close, vs SR=0) | {_pf(psr_pass)} | {psr_value:.4f} | > {PSR_THRESHOLD} |')
    lines.append(f'| 2. DSR(V12c, n_trials={n_trials_used}) | {_pf(dsr_pass)} | {dsr_value:.4f} | > {DSR_THRESHOLD} |')
    lines.append(
        f'| 3. PBO across {{V01,V04,V05,V06,V11,V12,V12c}} | {_pf(pbo_pass)} | '
        f'{pbo_value:.4f} | < {PBO_THRESHOLD} |'
    )
    lines.append(
        f'| 4. Lag-degradation (5 bps) | {_pf(lag_pass)} | '
        f'nc={nc_v12c_5:.3f}, lag={lag_v12c_5:.3f}, nc-lag={nc_minus_lag:+.3f} | '
        f'<= max(0.2*|nc|, 0.1) = {cap:.3f} |'
    )
    lines.append(
        f'| 5a. Sharpe(V12c @ 7.5bps lag) > 0.30 | {_pf(cost_floor_pass)} | '
        f'{v12c_lag_7p5:.4f} | > {COST_FLOOR_SHARPE} |'
    )
    lines.append(
        f'| 5b. Sharpe(V12c) >= 0.9 * Sharpe(V11) @ 7.5bps lag | {_pf(no_regress_pass)} | '
        f'V12c={v12c_lag_7p5:.4f}, 0.9*V11={COST_NO_REGRESS_FRACTION * v11_ref:.4f} '
        f'(V11={v11_ref:.4f}) | >= {COST_NO_REGRESS_FRACTION} * V11 |'
    )
    lines.append('')
    lines.append(f'**Overall tier**: {verdict_short} -- {verdict}')
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
    lines.append(f'| Observed Sharpe (V12c) | {psr_info["sr_daily"]:.6f} | {psr_info["sr_annual"]:.4f} |')
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

    lines.append('## Cost grid (V12c)')
    lines.append('')
    lines.append('| Cost bps | Mode | Sharpe | CAGR |')
    lines.append('|---:|:--|---:|---:|')
    for cb in COST_GRID_BPS:
        for mode in ('near_close', 'one_day_lag'):
            rr = cost_grid_results[(cb, mode)]
            lines.append(f'| {cb:.1f} | {mode} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    lines.append('## Cross-variants comparison (5 bps near_close, 7 variants)')
    lines.append('')
    lines.append('| Variant | Sharpe | CAGR |')
    lines.append('|---|---:|---:|')
    for v in CROSS_LABELS:
        rr = cross_runs_at_gate_cost[v]
        lines.append(f'| {v} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    lines.append('## PBO')
    lines.append('')
    lines.append(f'- Matrix shape: {matrix_shape[0]} x {matrix_shape[1]} '
                 f'({", ".join(CROSS_LABELS)})')
    lines.append(f'- s (CSCV submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_value:.4f}')
    if pbo_value < PBO_THRESHOLD:
        lines.append(f'- Interpretation: {pbo_value:.4f} < {PBO_THRESHOLD} = low overfitting evidence')
    else:
        lines.append(f'- Interpretation: {pbo_value:.4f} >= {PBO_THRESHOLD} = elevated overfitting risk')
    lines.append('')

    lines.append('## Detector-onset alignment panel (BEAR or UNPREDICTABLE)')
    lines.append('')
    events = alignment.get('events', [])
    if not events:
        lines.append('_No BEAR/UNPREDICTABLE onsets in the test window._')
    else:
        lines.append('Per-onset breakdown of V12c cash response. V12c goes to cash on '
                     'BOTH detector-BEAR and detector-UNPREDICTABLE onsets. `gap_days` = '
                     'trading-day gap between actual onset and the SPY drawdown trough '
                     'within [-20d, +30d]; positive = onset late vs trough. '
                     '`avoided_return` = SPY return during the cash window.')
        lines.append('')
        lines.append('| Onset | Regime | Window | Cash window | Cash days | SPY trough | Gap days | Avoided return |')
        lines.append('|---|---|---|---|---:|---|---:|---:|')
        for e in events:
            lines.append(
                f'| {e["onset"]} | {e["onset_regime"]} | '
                f'{e["window_start"]} .. {e["window_end"]} | '
                f'{e["cash_start"]} .. {e["cash_end"]} | {e["cash_days"]} | '
                f'{e["spy_trough_date"]} | {e["gap_days"]:+d} | '
                f'{_fmt_pct(e["avoided_return"])} |'
            )
        lines.append('')
        lines.append(f'**Aggregate gap days (mean)**: {alignment["aggregate_gap_days"]:+.2f}')
        lines.append(f'**Aggregate avoided return (mean)**: '
                     f'{_fmt_pct(alignment["aggregate_avoided_return"])}')
        lines.append(f'**Onset count**: {len(events)}')
    lines.append('')

    lines.append('## Sensitivity appendix -- COVID-excluded subgroup (E2 robustness)')
    lines.append('')
    lines.append('_Per analyst direction following E2 verdict AMBIGUOUS (53.6% attribution '
                 'in top-3 events, COVID-dominant). This panel is INFORMATIONAL ONLY '
                 'and does NOT influence the gate verdict per spec rev4 honesty '
                 'discipline; the gates stand on the full-window numbers._')
    lines.append('')
    lines.append(f'COVID exclusion window: {covid_panel["window_start"]} .. {covid_panel["window_end"]} (inclusive).')
    lines.append('')
    lines.append('| Metric | Full window | COVID-excluded | Delta |')
    lines.append('|---|---:|---:|---:|')
    lines.append(
        f'| Sharpe (V12c @ 5bps near_close) | {covid_panel["sharpe_full_window"]:.4f} | '
        f'{covid_panel["sharpe_ex_covid"]:.4f} | {covid_panel["sharpe_delta"]:+.4f} |'
    )
    lines.append(
        f'| CAGR | {_fmt_pct(covid_panel["cagr_full_window"])} | '
        f'{_fmt_pct(covid_panel["cagr_ex_covid"])} | '
        f'{(covid_panel["cagr_ex_covid"] - covid_panel["cagr_full_window"]) * 100:+.2f}pp |'
    )
    lines.append(
        f'| Sample days | {covid_panel["n_days_full"]} | '
        f'{covid_panel["n_days_ex_covid"]} | -{covid_panel["n_days_dropped"]} |'
    )
    lines.append('')
    covid_share_pct = abs(covid_panel['sharpe_delta']) / max(abs(covid_panel['sharpe_full_window']), 1e-9) * 100
    if covid_panel['sharpe_delta'] < -0.05:
        flag = (
            f'**[!] HONESTY FLAG**: removing the COVID window drops V12c Sharpe by '
            f'{covid_panel["sharpe_delta"]:+.4f} ({covid_share_pct:.1f}% of the full-window '
            'Sharpe in magnitude). The V12c edge is materially concentrated in the COVID '
            'event. Treat the full-window verdict with caution; deploy decision should '
            'account for this concentration.'
        )
    elif covid_panel['sharpe_delta'] > 0.05:
        flag = (
            f'**Note**: V12c Sharpe IMPROVES when COVID is excluded (+{covid_panel["sharpe_delta"]:.4f}). '
            'The full-window verdict is conservative: V12c does not depend on COVID for '
            'its measured edge.'
        )
    else:
        flag = (
            f'**Note**: Sharpe shift under COVID exclusion is small ({covid_panel["sharpe_delta"]:+.4f}, '
            f'{covid_share_pct:.1f}% of full-window magnitude). V12c edge is not '
            'concentrated in the COVID event.'
        )
    lines.append(flag)
    lines.append('')

    lines.append('## Methodology decisions')
    lines.append('')
    lines.append(f'- n_trials_project = {n_trials_used} ({n_trials_source})')
    lines.append(f'- PBO s = {PBO_S} (methodology Section 2.4 default)')
    lines.append(f'- Cost tier for PSR/DSR/PBO: {GATE_COST_BPS} bps per side')
    lines.append(
        '- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, '
        'trades executed at close T+1, MTM at close T+1.'
    )
    lines.append('- All metrics computed on net-of-cost daily returns.')
    lines.append('- Variants run at full window (start..end). Phase 4 is single-pass, '
                 'not walk-forward.')
    lines.append('- Gate 4 (rev4, directional): `(nc - lag) <= max(0.2 * |nc|, 0.1)` -- '
                 'lag > near_close is the safe direction and is not penalized.')
    lines.append('- Gate 5 (rev4-followup): both clauses required: '
                 'Sharpe(V12c @ 7.5bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5bps lag).')
    lines.append('- Tier classification: TIER 1 (all 5 pass) / TIER 3 (structural pass, '
                 'PSR+DSR fail) / TIER 4 (any structural fail).')
    lines.append('- COVID exclusion is post-hoc filter of the V12c@5bps-near_close record '
                 'stream (NOT a fresh backtest, so no PSR/DSR distortion).')
    lines.append('')

    lines.append('## Metadata')
    lines.append('')
    lines.append(f'- Git SHA: {sha}')
    lines.append(f'- Run datetime: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'- n_trials_project source: {n_trials_source}')
    lines.append(f'- V11 reference (Gate 5): inline re-run @ 7.5bps one_day_lag = {v11_ref:.4f}; '
                 f'V11-readiness-doc value = {V11_REF_SHARPE_AT_7P5BPS_LAG:.4f}')
    lines.append('- Total gate-influencing unique runs: 15 (8 V12c cost grid + 6 cross [V12c reused] + 1 V11 ref)')
    lines.append('- Sensitivity panels: 1 (COVID-excluded subgroup, post-hoc filter only)')
    lines.append('')

    return '\n'.join(lines)


def _smoke_run(args) -> int:
    """Run ONE V12c backtest @ 5bps near_close and exit."""
    logger.info('[+] Smoke mode: one V12c backtest @ 5bps near_close')
    t0 = time.time()
    rr = _run(args, 'V12c', 5.0, 'near_close')
    elapsed = time.time() - t0
    logger.info(f'[+] Smoke complete in {elapsed:.1f}s. Sharpe={rr.sharpe:.4f}, '
                f'CAGR={rr.cagr:.2%}, n_days={len(rr.records)}')
    return 0


def main() -> int:
    args = _parse_args()

    if args.smoke:
        return _smoke_run(args)

    logger.info(
        f'[+] Starting V12c readiness orchestrator (Experiment 6): '
        f'window {args.start.date()}..{args.end.date()}'
    )
    t_start = time.time()

    # ----- Cost grid (8 runs: V12c across 4 cost tiers x 2 timing modes) -----
    cost_grid: Dict[Tuple[float, str], RunResult] = {}
    for cb in COST_GRID_BPS:
        for mode in ('near_close', 'one_day_lag'):
            cost_grid[(cb, mode)] = _run(args, 'V12c', cb, mode)

    # ----- Cross-variants at 5 bps near_close (6 NEW runs; V12c reused from grid) -----
    cross_runs_at_gate_cost: Dict[str, RunResult] = {}
    cross_runs_at_gate_cost['V12c'] = cost_grid[(GATE_COST_BPS, 'near_close')]
    for v in CROSS_LABELS:
        if v == 'V12c':
            continue
        cross_runs_at_gate_cost[v] = _run(args, v, GATE_COST_BPS, 'near_close')

    # ----- V11 reference for Gate 5 (1 run) -----
    v11_lag_7p5 = _run(args, 'V11', 7.5, 'one_day_lag')

    logger.info('[+] All 15 unique runs complete. Computing PSR / DSR / PBO / COVID panel / alignment...')

    # ----- PBO across 7 variants at 5bps near_close -----
    records_by_label = {v: cross_runs_at_gate_cost[v].records for v in CROSS_LABELS}
    matrix = _build_pbo_matrix(records_by_label)
    pbo_value = pbo(matrix, s=PBO_S)
    logger.info(f'[+] PBO: {pbo_value:.4f} (matrix {matrix.shape})')

    # ----- PSR for V12c @ 5bps near_close -----
    v12c_5_nc = cost_grid[(GATE_COST_BPS, 'near_close')]
    psr_info = _compute_psr(v12c_5_nc.records)
    logger.info(f'[+] PSR (V12c vs SR=0): {psr_info["psr"]:.4f}')

    # ----- n_trials_project HARD-CODED at 23 -----
    n_trials_project = N_TRIALS_PROJECT
    n_trials_source = (
        'hard-coded 23: V12 used 22 (4 from experiments.duckdb + 18 V12 runs); '
        'V12c is trial #23, V12-up-cash sensitivity now formalized as its own gate'
    )
    logger.info(f'[+] n_trials_project = {n_trials_project} ({n_trials_source})')

    # ----- DSR -----
    trial_sharpes_annual = [cross_runs_at_gate_cost[v].sharpe for v in CROSS_LABELS]
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
            f'{v}={cross_runs_at_gate_cost[v].sharpe:.3f}' for v in CROSS_LABELS
        ),
        'v_sqrt_daily': v_sqrt,
    }
    logger.info(
        f'[+] DSR: prob={dsr_value:.4f} expected_max_annual={sr_zero_annual:.4f} '
        f'v_sqrt_daily={v_sqrt:.6f}'
    )

    # ----- COVID-excluded subgroup (post-hoc filter; informational only) -----
    covid_panel = _covid_excluded_sharpe(v12c_5_nc.records)
    logger.info(
        f'[+] COVID-excluded: full={covid_panel["sharpe_full_window"]:.4f}, '
        f'ex={covid_panel["sharpe_ex_covid"]:.4f}, '
        f'delta={covid_panel["sharpe_delta"]:+.4f} '
        f'(dropped {covid_panel["n_days_dropped"]} days)'
    )

    # ----- Detector-onset alignment panel (BEAR + UNPREDICTABLE onsets) -----
    panel = load_universe_panel(args.universe, args.start, args.end)
    alignment = _detector_onset_alignment(v12c_5_nc.records, panel)
    logger.info(f'[+] Alignment: {len(alignment["events"])} BEAR/UNPREDICTABLE onsets, '
                f'aggregate gap_days={alignment["aggregate_gap_days"]:+.2f}')

    # ----- Render report -----
    sha = _git_sha()
    md = _build_doc(
        args=args,
        sha=sha,
        cost_grid_results=cost_grid,
        cross_runs_at_gate_cost=cross_runs_at_gate_cost,
        v11_lag_7p5=v11_lag_7p5,
        covid_panel=covid_panel,
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
