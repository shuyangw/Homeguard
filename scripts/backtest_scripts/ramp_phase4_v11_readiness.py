#!/usr/bin/env python3
"""V11 Phase D readiness orchestrator.

Runs the 12-backtest battery required to gate V11 for paper deployment:
  - V01/V04/V05/V06/V11 at 5 bps near_close (5 runs)
  - V11 at [0.0, 2.5, 7.5] bps near_close (3 runs)
  - V11 at [0.0, 2.5, 5.0, 7.5] bps one_day_lag (4 runs)

Emits PSR (vs SR=0), DSR (n_trials=20), PBO (s=16) on the {V01,V04,V05,V06,V11}
cross-section, and a near_close-vs-one_day_lag robustness sweep for V11.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.backtesting.statistics.pbo import pbo
from src.backtesting.statistics.psr import psr
from src.backtesting.statistics.dsr import dsr, expected_max_sharpe
from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import cagr, sharpe_ratio
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import get_logger


logger = get_logger(__name__)

DELTA_REBALANCE_PCT_BY_VARIANT = {
    'V06': 0.02,
    'V11': 0.02,
}

CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11')
COST_TIERS = (0.0, 2.5, 5.0, 7.5)
GATE_COST_BPS = 5.0
N_TRIALS = 20
PBO_S = 16
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
LAG_DELTA_TOLERANCE_PCT = 20.0


@dataclass
class RunResult:
    variant_id: str
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
    p = argparse.ArgumentParser(description='V11 Phase D readiness orchestrator.')
    p.add_argument('--smoke', action='store_true', help='3-year slice instead of full window (enough for regime detector warmup).')
    p.add_argument('--start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'), default=datetime(2017, 1, 1))
    p.add_argument('--end', type=lambda s: datetime.strptime(s, '%Y-%m-%d'), default=datetime(2026, 5, 16))
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path, default=Path('docs/reports/ramp/20260523_phase4_v11_readiness.md'))
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


def _run(args, variant_id: str, cost_bps: float, timing_mode: str) -> RunResult:
    cfg = _build_cfg(args, variant_id, cost_bps, timing_mode)
    spec = REGISTRY[variant_id]
    logger.info(f'[+] Running {variant_id} {timing_mode} cost={cost_bps}bps...')
    t0 = time.time()
    records = run_variant(cfg, spec)
    elapsed = time.time() - t0
    if not records:
        raise RuntimeError(f'{variant_id} {timing_mode} {cost_bps}bps produced no records')
    rets = pd.Series([r.daily_return for r in records])
    eq = (1 + rets).cumprod() * args.initial_capital
    sr = sharpe_ratio(rets)
    cg = cagr(eq)
    logger.info(
        f'[+] {variant_id} {timing_mode} {cost_bps}bps done in {elapsed:.1f}s: '
        f'Sharpe={sr:.3f} CAGR={cg:.2%} n_days={len(records)}'
    )
    return RunResult(variant_id=variant_id, cost_bps=cost_bps, timing_mode=timing_mode,
                     records=records, sharpe=sr, cagr=cg)


def _build_pbo_matrix(records_by_variant: Dict[str, List[DailyRecord]]) -> np.ndarray:
    dates_v01 = [r.date for r in records_by_variant['V01']]
    for v in CROSS_VARIANTS[1:]:
        dates_v = [r.date for r in records_by_variant[v]]
        if dates_v != dates_v01:
            raise RuntimeError(
                f'Date mismatch: V01 has {len(dates_v01)} dates, {v} has {len(dates_v)}'
            )
    matrix = np.column_stack([
        np.array([r.daily_return for r in records_by_variant[v]], dtype=np.float64)
        for v in CROSS_VARIANTS
    ])
    return matrix


def _compute_psr_v11(v11_records: List[DailyRecord]) -> Dict[str, float]:
    rets = np.array([r.daily_return for r in v11_records], dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 or rets.std() == 0:
        raise RuntimeError('V11 returns are degenerate; cannot compute PSR.')
    sr_daily = rets.mean() / rets.std()
    sr_annual = sr_daily * np.sqrt(252)
    skew = float(np.mean(((rets - rets.mean()) / rets.std()) ** 3))
    pearson_kurt = float(np.mean(((rets - rets.mean()) / rets.std()) ** 4))
    psr_val = psr(sr_hat=sr_annual, sr_benchmark=0.0, n=len(rets), skew=skew, kurt=pearson_kurt)
    return {
        'sr_annual': float(sr_annual),
        'skew': skew,
        'pearson_kurt': pearson_kurt,
        'n_days': int(len(rets)),
        'psr': float(psr_val),
    }


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.2f}%'


def _build_doc(
    args,
    sha: str,
    cross_runs: Dict[str, RunResult],
    near_close_v11_by_tier: Dict[float, RunResult],
    one_day_lag_v11_by_tier: Dict[float, RunResult],
    psr_info: Dict[str, float],
    dsr_info: Dict[str, float],
    pbo_value: float,
    matrix_shape: tuple,
) -> str:
    psr_value = psr_info['psr']
    psr_pass = psr_value > PSR_THRESHOLD
    dsr_value = dsr_info['dsr']
    dsr_pass = dsr_value > DSR_THRESHOLD
    pbo_pass = pbo_value < PBO_THRESHOLD

    nc_5 = near_close_v11_by_tier[5.0].sharpe
    lag_5 = one_day_lag_v11_by_tier[5.0].sharpe
    if nc_5 == 0:
        delta_pct = float('inf')
    else:
        delta_pct = (lag_5 - nc_5) / abs(nc_5) * 100.0
    lag_pass = abs(delta_pct) <= LAG_DELTA_TOLERANCE_PCT

    overall_pass = psr_pass and dsr_pass and pbo_pass and lag_pass
    verdict = 'READY for Phase D paper deploy' if overall_pass else 'BLOCKED -- fall back to V05'

    def _pf(b: bool) -> str:
        return 'PASS' if b else 'FAIL'

    lines: List[str] = []
    lines.append(f'# V11 Phase D Readiness -- 2026-05-23')
    lines.append('')
    lines.append(f'**Code commit**: {sha}')
    lines.append(f'**Data**: Alpaca SIP daily-aggregated, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Cost tier for gates**: {GATE_COST_BPS} bps per side')
    lines.append(f'**n_trials**: {N_TRIALS} (conservative; covers Phase 4 + Phase 3 variant grid)')
    lines.append('')
    lines.append('## Verdict')
    lines.append('')
    lines.append('| Gate | Result | Value | Threshold |')
    lines.append('|---|:---:|---:|---:|')
    lines.append(f'| PSR (vs SR=0) | {_pf(psr_pass)} | {psr_value:.4f} | > {PSR_THRESHOLD} |')
    lines.append(f'| DSR (n_trials={N_TRIALS}) | {_pf(dsr_pass)} | {dsr_value:.4f} | > {DSR_THRESHOLD} |')
    lines.append(f'| PBO across {{V01,V04,V05,V06,V11}} | {_pf(pbo_pass)} | {pbo_value:.4f} | < {PBO_THRESHOLD} |')
    lines.append(
        f'| One-day-lag Sharpe robustness (5 bps) | {_pf(lag_pass)} | '
        f'nc={nc_5:.3f}, lag={lag_5:.3f}, delta={delta_pct:+.2f}% | within {LAG_DELTA_TOLERANCE_PCT:.0f}% |'
    )
    lines.append('')
    lines.append(f'**Overall**: {verdict}')
    lines.append('')

    lines.append('## PSR / DSR detail')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    lines.append(f'| Observed annualized Sharpe | {psr_info["sr_annual"]:.4f} |')
    lines.append(
        f'| Expected max Sharpe under null (n_trials={N_TRIALS}, scaled by V[trial_sharpes]) | '
        f'{dsr_info["expected_max_sharpe"]:.4f} |'
    )
    lines.append(f'| DSR probability (true SR > expected max) | {dsr_value:.4f} |')
    lines.append(f'| PSR (vs SR=0) | {psr_value:.4f} |')
    lines.append(f'| Trial Sharpes used for V[trial] term | {dsr_info["trial_sharpes_str"]} |')
    lines.append(f'| sqrt(V[trial_sharpes]) | {dsr_info["v_sqrt"]:.4f} |')
    lines.append(f'| Skewness | {psr_info["skew"]:.4f} |')
    lines.append(f'| Pearson kurtosis | {psr_info["pearson_kurt"]:.4f} |')
    lines.append(f'| Sample size (days) | {psr_info["n_days"]} |')
    lines.append('')

    lines.append('## PBO')
    lines.append('')
    lines.append(f'- Matrix shape: {matrix_shape[0]} x {matrix_shape[1]} (V01, V04, V05, V06, V11)')
    lines.append(f'- s (submatrices): {PBO_S}')
    lines.append(f'- PBO value: {pbo_value:.4f}')
    lines.append(f'- Interpretation: {pbo_value:.4f} < {PBO_THRESHOLD} is low overfitting evidence')
    lines.append('')

    lines.append('### Per-variant Sharpe at 5 bps near_close')
    lines.append('')
    lines.append('| Variant | Sharpe | CAGR |')
    lines.append('|---|---:|---:|')
    for v in CROSS_VARIANTS:
        rr = cross_runs[v]
        lines.append(f'| {v} | {rr.sharpe:.4f} | {_fmt_pct(rr.cagr)} |')
    lines.append('')

    lines.append('## One-day-lag robustness sweep (V11)')
    lines.append('')
    lines.append('| Cost bps | near_close Sharpe | one_day_lag Sharpe | Delta % |')
    lines.append('|---|---:|---:|---:|')
    for tier in COST_TIERS:
        nc = near_close_v11_by_tier[tier].sharpe
        lg = one_day_lag_v11_by_tier[tier].sharpe
        if nc == 0:
            dp = float('inf')
            dp_str = 'inf'
        else:
            dp = (lg - nc) / abs(nc) * 100.0
            dp_str = f'{dp:+.2f}%'
        lines.append(f'| {tier:.1f} | {nc:.4f} | {lg:.4f} | {dp_str} |')
    lines.append('')
    lines.append(
        'If one_day_lag Sharpes at 5 bps degrade by more than 20% from near_close, '
        'V11 has structural lookahead.'
    )
    lines.append('')

    lines.append('## Methodology decisions')
    lines.append('')
    lines.append(f'- n_trials = {N_TRIALS} (conservative; methodology Section 2.3 cumulative trial count)')
    lines.append(f'- PBO s = {PBO_S} (methodology Section 2.4 default)')
    lines.append(f'- Cost tier for PSR/DSR/PBO: {GATE_COST_BPS} bps per side')
    lines.append(
        '- one_day_lag definition: signal computed at close T from `panel.loc[:T]`, '
        'trades executed at close T+1, MTM at close T+1'
    )
    lines.append('- All metrics computed on net-of-cost daily returns')
    lines.append(
        '- Variants run at full window (start..end). No purge/embargo applied within-variant '
        '(Phase 4 is single-pass, not walk-forward).'
    )
    lines.append('')

    lines.append('## What happens next')
    lines.append('')
    if overall_pass:
        lines.append(
            'READY: extend the A7 paper-validation comparator '
            '(`scripts/trading/compare_paper_vs_plan.py`) to model V11\'s rank_buffer + '
            'min_hold + delta_threshold filter state. Then enable V11 in production paper '
            'trading on EC2 per the A7 discipline (4-6 weeks paper validation).'
        )
    else:
        lines.append(
            'BLOCKED: re-run this orchestrator on V05 (single-line change: replace `V11` with '
            '`V05` in the gate-target variable). V05 was documented in Wave 1 as the safe '
            'fallback (95% of V11\'s edge, one fewer moving part).'
        )
    lines.append('')

    return '\n'.join(lines)


def main() -> int:
    args = _parse_args()
    if args.smoke:
        args.end = args.start + timedelta(days=365 * 3)
        logger.info(f'[+] Smoke mode: window narrowed to {args.start.date()}..{args.end.date()}')

    logger.info(
        f'[+] Starting V11 readiness orchestrator: window {args.start.date()}..{args.end.date()}'
    )
    t_start = time.time()

    cross_runs: Dict[str, RunResult] = {}
    for v in CROSS_VARIANTS:
        cross_runs[v] = _run(args, v, GATE_COST_BPS, 'near_close')

    near_close_v11_by_tier: Dict[float, RunResult] = {}
    near_close_v11_by_tier[GATE_COST_BPS] = cross_runs['V11']
    for tier in COST_TIERS:
        if tier == GATE_COST_BPS:
            continue
        near_close_v11_by_tier[tier] = _run(args, 'V11', tier, 'near_close')

    one_day_lag_v11_by_tier: Dict[float, RunResult] = {}
    for tier in COST_TIERS:
        one_day_lag_v11_by_tier[tier] = _run(args, 'V11', tier, 'one_day_lag')

    logger.info('[+] All 12 runs complete. Computing PSR / DSR / PBO...')

    records_by_variant = {v: cross_runs[v].records for v in CROSS_VARIANTS}
    matrix = _build_pbo_matrix(records_by_variant)
    pbo_value = pbo(matrix, s=PBO_S)
    logger.info(f'[+] PBO: {pbo_value:.4f} (matrix {matrix.shape})')

    psr_info = _compute_psr_v11(cross_runs['V11'].records)
    logger.info(f'[+] PSR (V11 vs SR=0): {psr_info["psr"]:.4f}')

    # DSR per methodology Section 2.3: use the 5 Phase 4 variants' Sharpes
    # to estimate V[trial_sharpes], with n_trials_project=20 as the
    # conservative cumulative-trials count.
    trial_sharpes = [cross_runs[v].sharpe for v in CROSS_VARIANTS]
    sr_zero = expected_max_sharpe(trial_sharpes, N_TRIALS)
    dsr_value = dsr(
        sr_hat=psr_info['sr_annual'],
        trial_sharpes=trial_sharpes,
        n=psr_info['n_days'],
        skew=psr_info['skew'],
        kurt=psr_info['pearson_kurt'],
        n_trials_project=N_TRIALS,
    )
    v_sqrt = float(np.sqrt(np.var(trial_sharpes, ddof=1)))
    dsr_info = {
        'dsr': float(dsr_value),
        'expected_max_sharpe': float(sr_zero),
        'trial_sharpes_str': ', '.join(f'{v}={cross_runs[v].sharpe:.3f}' for v in CROSS_VARIANTS),
        'v_sqrt': v_sqrt,
    }
    logger.info(
        f'[+] DSR: prob={dsr_value:.4f} expected_max={sr_zero:.4f} '
        f'v_sqrt={v_sqrt:.4f} (threshold > 0.95)'
    )

    sha = _git_sha()
    md = _build_doc(
        args=args,
        sha=sha,
        cross_runs=cross_runs,
        near_close_v11_by_tier=near_close_v11_by_tier,
        one_day_lag_v11_by_tier=one_day_lag_v11_by_tier,
        psr_info=psr_info,
        dsr_info=dsr_info,
        pbo_value=pbo_value,
        matrix_shape=matrix.shape,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    total_elapsed = time.time() - t_start
    logger.info(f'[+] Wrote {args.output}')
    logger.info(f'[+] Total wall-clock: {total_elapsed / 60:.2f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
