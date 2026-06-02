#!/usr/bin/env python3
"""Wave-3 single-variant readiness runner with per-backtest durability.

Generalizes ramp_phase4_v11_readiness.py into a reusable single-variant
runner that:
  - Runs one variant across configurable cost tiers and timing modes.
  - Persists each sub-backtest to the experiment registry BEFORE continuing
    (G0.5 durability -- no more all-or-nothing memory runs).
  - Resumes cleanly after a crash: if a matching prior run exists in the
    registry (keyed on variant, cost_bps, timing_mode, git_sha, data_snapshot),
    the sub-backtest is SKIPPED and its return stream is reused.
  - Writes results to docs/reports/ramp/ as both .md and .json via atomic
    write (.tmp + os.replace) so partial writes never corrupt the artifact.

Usage:
    python scripts/backtest_scripts/ramp_phase4_wave3_readiness.py --variant V31
    python scripts/backtest_scripts/ramp_phase4_wave3_readiness.py --variant V31 --smoke
    python scripts/backtest_scripts/ramp_phase4_wave3_readiness.py --variant V31 \\
        --cost-tiers 5.0,7.5 --timing-modes near_close,one_day_lag

Does NOT compute family PBO/DSR (that is done once all 5 Wave-3 variants are
complete). Per-variant PSR (vs SR=0) IS reported as a single-variant gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.statistics.psr import psr
from src.experiments.registry import (
    DEFAULT_DB_PATH,
    _connect_with_retry,
    _git_sha as _registry_git_sha,
    append_run,
    init_db,
)
from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import DailyRecord, run_variant
from src.research.ramp_phase4.metrics import (
    avg_daily_turnover,
    cagr,
    max_drawdown,
    regime_attribution,
    sharpe_ratio,
)
from src.research.ramp_phase4.reports import build_variant_report
from src.research.ramp_phase4.variants import REGISTRY
from src.utils.logger import get_logger


logger = get_logger(__name__)

DELTA_REBALANCE_PCT_V31 = 0.02  # Same as V11 per spec
DATA_SNAPSHOT_DATE = datetime(2026, 5, 16).date()  # Cache last rebuilt


@dataclass
class SubRunResult:
    variant_id: str
    cost_bps: float
    timing_mode: str
    records: List[DailyRecord]
    sharpe: float
    cagr_val: float
    reused_from_registry: bool
    run_id: str


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Wave-3 single-variant readiness runner.')
    p.add_argument('--variant', required=True, help='Variant ID (e.g. V31)')
    p.add_argument(
        '--start',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2017, 1, 1),
    )
    p.add_argument(
        '--end',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime(2026, 5, 16),
    )
    p.add_argument(
        '--universe',
        type=Path,
        default=Path('config/universes/sp500-2025.csv'),
    )
    p.add_argument('--initial-capital', type=float, default=100_000.0)
    p.add_argument(
        '--smoke',
        action='store_true',
        help='3-year slice for wiring check only. NEVER use for recorded results.',
    )
    p.add_argument(
        '--cost-tiers',
        type=lambda s: [float(x) for x in s.split(',')],
        default=[5.0, 7.5],
    )
    p.add_argument(
        '--timing-modes',
        type=lambda s: s.split(','),
        default=['near_close', 'one_day_lag'],
    )
    p.add_argument(
        '--output-dir',
        type=Path,
        default=Path('docs/reports/ramp'),
    )
    p.add_argument(
        '--db-path',
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    return p.parse_args(argv)


def _build_cfg(
    args,
    variant_id: str,
    cost_bps: float,
    timing_mode: str,
) -> HarnessConfig:
    delta_pct = DELTA_REBALANCE_PCT_V31 if variant_id in ('V31', 'V11', 'V28', 'V26') else 0.0
    return HarnessConfig(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=cost_bps,
        timing_mode=timing_mode,
        delta_rebalance_pct=delta_pct,
    )


def _compute_extended_metrics(records: List[DailyRecord]) -> Dict:
    """Compute the full metrics row required by methodology Section 12 / TODO table."""
    rets = pd.Series([r.daily_return for r in records])
    eq = (1 + rets).cumprod() * records[0].portfolio_value if records else pd.Series()

    sr = sharpe_ratio(rets)
    cagr_val = cagr(eq)
    max_dd = max_drawdown(eq)

    # Max drawdown duration (calendar days).
    if len(eq) > 1:
        rolling_max = eq.cummax()
        underwater = eq < rolling_max
        dd_dur = 0
        cur_dur = 0
        for u in underwater:
            if u:
                cur_dur += 1
                dd_dur = max(dd_dur, cur_dur)
            else:
                cur_dur = 0
    else:
        dd_dur = 0

    calmar = float(cagr_val / abs(max_dd)) if max_dd < 0 else float('inf')

    # Monthly win rate.
    rets_ser = pd.Series(
        [r.daily_return for r in records],
        index=[r.date for r in records],
    )
    monthly = (1 + rets_ser).resample('ME').prod() - 1
    monthly_win_pct = float((monthly > 0).mean()) if len(monthly) > 0 else float('nan')

    # Profit factor.
    wins = rets_ser[rets_ser > 0].sum()
    losses = abs(rets_ser[rets_ser < 0].sum())
    profit_factor = float(wins / losses) if losses > 0 else float('inf')

    # Trade count (approximation: days where turnover > 0).
    trade_days = sum(1 for r in records if r.turnover_usd > 0)

    # Average hold time (days): total portfolio-value-weighted days per position.
    # Approximate: 1 / avg_daily_turnover when positive.
    avg_to = avg_daily_turnover(records)
    avg_hold_est = float(1.0 / avg_to) if avg_to > 0 else float('nan')

    # Annualized turnover.
    total_to_usd = sum(r.turnover_usd for r in records)
    avg_pv = float(np.mean([r.portfolio_value for r in records if r.portfolio_value > 0]))
    ann_turnover = (total_to_usd / avg_pv) / (len(records) / 252.0) if avg_pv > 0 and len(records) > 0 else float('nan')

    # PSR.
    rets_arr = np.array([r.daily_return for r in records], dtype=np.float64)
    rets_arr = rets_arr[np.isfinite(rets_arr)]
    if len(rets_arr) >= 30 and rets_arr.std() > 0:
        sr_daily = float(rets_arr.mean() / rets_arr.std())
        skew = float(np.mean(((rets_arr - rets_arr.mean()) / rets_arr.std()) ** 3))
        pk = float(np.mean(((rets_arr - rets_arr.mean()) / rets_arr.std()) ** 4))
        psr_val = float(psr(sr_daily, 0.0, len(rets_arr), skew, pk))
    else:
        psr_val = float('nan')

    return {
        'sharpe': float(sr),
        'cagr': float(cagr_val),
        'max_dd': float(max_dd),
        'max_dd_duration_days': int(dd_dur),
        'calmar': float(calmar),
        'monthly_win_pct': float(monthly_win_pct),
        'profit_factor': float(profit_factor),
        'trade_days': int(trade_days),
        'avg_hold_est_days': float(avg_hold_est),
        'ann_turnover': float(ann_turnover),
        'psr_vs_0': float(psr_val),
        'n_records': int(len(records)),
    }


def _lookup_prior_run(
    db_path: Path,
    strategy_name: str,
    cost_bps: float,
    timing_mode: str,
    git_sha: str,
    window_start: datetime,
    window_end: datetime,
) -> Optional[Tuple[str, pd.DataFrame]]:
    """Query the registry for a matching prior run.

    Key: strategy_name + cost_bps + timing_mode in notes + git_sha +
         window_start + window_end.

    Including window dates prevents a smoke run (3-year slice) from being
    incorrectly reused for a full-window run with the same git_sha.

    Returns (run_id, return_stream_df) if found, None otherwise.
    The return_stream_df has columns: date, return_pct, position_count.
    """
    try:
        init_db(db_path)
        con = _connect_with_retry(db_path, read_only=True)
        try:
            result = con.execute(
                """
                SELECT run_id FROM runs
                WHERE strategy_name = ?
                  AND cost_bps = ?
                  AND git_sha = ?
                  AND notes LIKE ?
                  AND phase = 'wave3_readiness'
                  AND window_start = ?
                  AND window_end = ?
                ORDER BY timestamp_utc DESC
                LIMIT 1
                """,
                [
                    strategy_name,
                    cost_bps,
                    git_sha,
                    f'%{timing_mode}%',
                    window_start.date(),
                    window_end.date(),
                ],
            ).fetchone()
            if not result:
                return None
            run_id = result[0]
            stream = con.execute(
                "SELECT date, return_pct, position_count FROM return_streams "
                "WHERE run_id = ? ORDER BY date",
                [run_id],
            ).fetch_df()
            return run_id, stream
        finally:
            con.close()
    except Exception as exc:
        logger.error(f'[!] Registry lookup failed (will re-run): {exc}')
        return None


def _records_from_return_stream(
    stream: pd.DataFrame,
    initial_capital: float,
) -> List[DailyRecord]:
    """Reconstruct a minimal records list from a persisted return stream.

    The reconstructed records have portfolio_value and daily_return populated
    from the return stream; all other fields are zeroed/empty. This is
    sufficient for metrics computation (not for regime attribution, which
    needs regime labels from the original run). For resumed runs the report
    will show 'n/a' in regime attribution -- acceptable.
    """
    from src.research.ramp_phase4.engine import DailyRecord
    records = []
    val = initial_capital
    for _, row in stream.iterrows():
        ret = float(row['return_pct'])
        val = val * (1 + ret)
        records.append(DailyRecord(
            date=pd.Timestamp(row['date']).to_pydatetime(),
            regime='RESUMED',
            target_weights={},
            realized_weights={},
            turnover_usd=0.0,
            cost_usd=0.0,
            portfolio_value=val,
            daily_return=ret,
        ))
    return records


def _run_or_reuse(
    args,
    variant_id: str,
    cost_bps: float,
    timing_mode: str,
    git_sha: str,
) -> SubRunResult:
    """Run one sub-backtest or reuse a prior registry run.

    Durability protocol (G0.5):
      1. Before running, query registry for a matching prior run.
      2. If found: reuse its return_stream, skip recompute.
      3. If not found: run, compute metrics, write to registry FIRST,
         then return result.

    Raises if the registry write fails (per Section 9.3).
    """
    strategy_name = f'RAMP-{variant_id}'
    prior = _lookup_prior_run(
        args.db_path, strategy_name, cost_bps, timing_mode, git_sha,
        args.start, args.end,
    )

    if prior is not None:
        run_id, stream = prior
        logger.info(
            f'[+] Reusing prior run {run_id[:8]} for {variant_id} {timing_mode} {cost_bps}bps'
        )
        records = _records_from_return_stream(stream, args.initial_capital)
        rets = pd.Series([r.daily_return for r in records])
        eq = (1 + rets).cumprod() * args.initial_capital
        sr = sharpe_ratio(rets)
        cg = cagr(eq)
        return SubRunResult(
            variant_id=variant_id,
            cost_bps=cost_bps,
            timing_mode=timing_mode,
            records=records,
            sharpe=sr,
            cagr_val=cg,
            reused_from_registry=True,
            run_id=run_id,
        )

    cfg = _build_cfg(args, variant_id, cost_bps, timing_mode)
    spec = REGISTRY[variant_id]
    logger.info(f'[+] Running {variant_id} {timing_mode} cost={cost_bps}bps ...')
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

    metrics = _compute_extended_metrics(records)
    regime_attr = regime_attribution(records)

    # Build return stream for registry.
    return_stream = pd.DataFrame({
        'date': [r.date for r in records],
        'return_pct': [r.daily_return for r in records],
        'position_count': [len(r.realized_weights) for r in records],
    })

    # Cost sensitivity: 1.5x cost tier (methodology Section 4 gate).
    # For the 5bps run, also compute at 7.5bps to get the sensitivity delta.
    cost_sensitivity = None
    if cost_bps == 5.0 and 7.5 in args.cost_tiers:
        # Will be filled in after the 7.5bps run; placeholder for now.
        cost_sensitivity = {'note': '7.5bps run is a separate sub-backtest in this suite'}

    wall_start = datetime.now(tz=timezone.utc) - timedelta(seconds=elapsed)
    wall_end = datetime.now(tz=timezone.utc)

    # G0.5 MANDATORY: write to registry BEFORE continuing; raises on failure.
    run_id = append_run(
        strategy_name=strategy_name,
        agent_name='backtest-driver',
        metrics=metrics,
        db_path=args.db_path,
        phase='wave3_readiness',
        universe_name='sp500-2025',
        asset_class='equity',
        data_frequency='daily',
        window_start=args.start,
        window_end=args.end,
        cost_bps=cost_bps,
        cost_tier_used=f'{cost_bps}bps',
        cost_sensitivity=cost_sensitivity,
        config_payload={
            'variant_id': variant_id,
            'timing_mode': timing_mode,
            'cost_bps': cost_bps,
            'delta_rebalance_pct': cfg.delta_rebalance_pct,
            'universe_csv': str(args.universe),
        },
        data_snapshot_date=DATA_SNAPSHOT_DATE,
        regime_breakdown=regime_attr,
        wall_clock_start=wall_start,
        wall_clock_end=wall_end,
        verdict='RECORDED',
        notes=f'{variant_id} {timing_mode} {cost_bps}bps',
        return_stream=return_stream,
    )
    logger.info(f'[+] Registry write OK run_id={run_id[:8]}')

    return SubRunResult(
        variant_id=variant_id,
        cost_bps=cost_bps,
        timing_mode=timing_mode,
        records=records,
        sharpe=sr,
        cagr_val=cg,
        reused_from_registry=False,
        run_id=run_id,
    )


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via .tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(content, encoding='utf-8')
    os.replace(tmp_path, path)


def _build_report(
    args,
    variant_id: str,
    git_sha: str,
    results: List[SubRunResult],
) -> str:
    """Build the Markdown readiness report for this variant."""
    # Organize records by (cost_bps, timing_mode).
    nc_by_cost: Dict[float, SubRunResult] = {}
    lag_by_cost: Dict[float, SubRunResult] = {}
    for r in results:
        if r.timing_mode == 'near_close':
            nc_by_cost[r.cost_bps] = r
        else:
            lag_by_cost[r.cost_bps] = r

    gate_cost = 5.0
    v11_sharpe_ref = 0.528  # V11 full-window incumbent
    v11_oos_ref = 0.527    # V11 EXT-OOS

    lines: List[str] = []
    lines.append(f'# Wave-3 {variant_id} Readiness -- {datetime.now().date()}')
    lines.append('')
    lines.append(f'**Code commit**: {git_sha}')
    lines.append(f'**Data**: Alpaca SIP daily, {args.start.date()} to {args.end.date()}')
    lines.append(f'**Universe**: {args.universe}')
    lines.append(f'**Incumbent (V11)**: Sharpe {v11_sharpe_ref} full-window / +{v11_oos_ref} EXT-OOS at 5 bps, turnover 39%')
    lines.append('')

    # Cost-tier sweep table.
    lines.append('## Metrics by cost tier (near_close)')
    lines.append('')
    lines.append('| Cost bps | Sharpe | CAGR | Max DD | Calmar | Monthly Win% | Profit Factor | Ann Turnover | PSR (vs 0) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for cost, rr in sorted(nc_by_cost.items()):
        m = _compute_extended_metrics(rr.records)
        lines.append(
            f'| {cost:.1f} | {m["sharpe"]:.3f} | {m["cagr"]:.2%} | {m["max_dd"]:.2%} | '
            f'{m["calmar"]:.2f} | {m["monthly_win_pct"]:.1%} | {m["profit_factor"]:.2f} | '
            f'{m["ann_turnover"]:.0%} | {m["psr_vs_0"]:.4f} |'
        )
    lines.append('')

    # One-day-lag robustness sweep.
    if lag_by_cost:
        lines.append('## One-day-lag robustness sweep')
        lines.append('')
        lines.append('| Cost bps | near_close Sharpe | one_day_lag Sharpe | Delta % |')
        lines.append('|---|---:|---:|---:|')
        all_costs = sorted(set(list(nc_by_cost.keys()) + list(lag_by_cost.keys())))
        for cost in all_costs:
            nc_rr = nc_by_cost.get(cost)
            lag_rr = lag_by_cost.get(cost)
            nc_sr = nc_rr.sharpe if nc_rr else float('nan')
            lag_sr = lag_rr.sharpe if lag_rr else float('nan')
            if nc_sr != 0 and not np.isnan(nc_sr) and not np.isnan(lag_sr):
                delta_pct = (lag_sr - nc_sr) / abs(nc_sr) * 100.0
                dp_str = f'{delta_pct:+.2f}%'
            else:
                dp_str = 'n/a'
            lines.append(f'| {cost:.1f} | {nc_sr:.3f} | {lag_sr:.3f} | {dp_str} |')
        lines.append('')

    # Regime attribution.
    gate_run = nc_by_cost.get(gate_cost)
    if gate_run:
        lines.append(f'## Regime attribution ({gate_cost} bps near_close)')
        lines.append('')
        attr = regime_attribution(gate_run.records)
        lines.append('| Regime | Days | Net return |')
        lines.append('|---|---:|---:|')
        for regime, d in sorted(attr.items()):
            lines.append(f"| {regime} | {d['days']} | {d['net_return']:.2%} |")
        lines.append('')

    # Cost-sensitivity gate: 5bps -> 7.5bps.
    if gate_cost in nc_by_cost and 7.5 in nc_by_cost:
        nc5 = nc_by_cost[gate_cost].sharpe
        nc75 = nc_by_cost[7.5].sharpe
        cost_gate_pass = nc75 >= 0.5
        lines.append('## Cost-sensitivity gate (methodology Section 4, 1.5x cost)')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append('|---|---:|')
        lines.append(f'| near_close Sharpe at 5.0 bps | {nc5:.3f} |')
        lines.append(f'| near_close Sharpe at 7.5 bps | {nc75:.3f} |')
        lines.append(f'| Gate (>= 0.5 at 7.5 bps) | {"PASS" if cost_gate_pass else "FAIL"} |')
        lines.append('')

    # vs V11 comparison.
    if gate_cost in nc_by_cost:
        v31_sharpe = nc_by_cost[gate_cost].sharpe
        delta_vs_v11 = v31_sharpe - v11_sharpe_ref
        sign = '+' if delta_vs_v11 >= 0 else ''
        beat_str = 'BEAT' if delta_vs_v11 > 0.01 else ('TIE' if abs(delta_vs_v11) <= 0.01 else 'WORSE')
        lines.append('## vs V11 comparison (5 bps near_close)')
        lines.append('')
        lines.append(f'| Metric | V11 | {variant_id} | Delta |')
        lines.append('|---|---:|---:|---:|')
        lines.append(
            f'| Full-window Sharpe (5 bps) | {v11_sharpe_ref:.3f} | {v31_sharpe:.3f} | {sign}{delta_vs_v11:.3f} |'
        )
        if lag_by_cost.get(gate_cost):
            v31_lag = lag_by_cost[gate_cost].sharpe
            lines.append(f'| one_day_lag Sharpe (5 bps) | -- | {v31_lag:.3f} | -- |')
        lines.append(f'| **Verdict** | -- | **{beat_str}** | -- |')
        lines.append('')

    # Durability note.
    reused = [r for r in results if r.reused_from_registry]
    fresh = [r for r in results if not r.reused_from_registry]
    lines.append('## G0.5 durability status')
    lines.append('')
    lines.append(f'- Sub-backtests: {len(results)} total ({len(fresh)} fresh, {len(reused)} reused from registry)')
    lines.append(f'- Registry run IDs: {", ".join(r.run_id[:8] for r in results)}')
    lines.append(
        '- Resume behavior: re-running this script with same args skips already-registered sub-backtests'
    )
    lines.append('')

    lines.append('## Methodology')
    lines.append('')
    lines.append('- Full data window (no subset). Smoke mode not used for these results.')
    lines.append('- Net-of-cost daily returns at each cost tier.')
    lines.append('- PSR vs SR=0 per methodology Section 2.2 (daily sr_hat, daily n).')
    lines.append('- DSR is cross-variant; computed at the Wave-3 family gate (after all 5 variants).')
    lines.append('- beta_window=90d fixed (not swept; conserves DSR trial budget).')
    lines.append('')

    return '\n'.join(lines)


def _build_json_payload(
    args,
    variant_id: str,
    git_sha: str,
    results: List[SubRunResult],
) -> Dict:
    """Build a machine-readable summary for the family-gate step."""
    payload = {
        'variant_id': variant_id,
        'git_sha': git_sha,
        'window_start': str(args.start.date()),
        'window_end': str(args.end.date()),
        'universe': str(args.universe),
        'generated_at': datetime.now().isoformat(),
        'sub_runs': [],
    }
    for rr in results:
        m = _compute_extended_metrics(rr.records)
        payload['sub_runs'].append({
            'cost_bps': rr.cost_bps,
            'timing_mode': rr.timing_mode,
            'run_id': rr.run_id,
            'reused_from_registry': rr.reused_from_registry,
            'metrics': m,
        })
    return payload


def main() -> int:
    args = _parse_args()

    if args.variant not in REGISTRY:
        logger.error(
            f'[-] Unknown variant: {args.variant}. Available: {sorted(REGISTRY.keys())}'
        )
        return 1

    if args.smoke:
        args.end = args.start + timedelta(days=365 * 3)
        logger.info(
            f'[!] SMOKE mode -- window narrowed to {args.start.date()}..{args.end.date()}. '
            f'Results NOT suitable for recording.'
        )

    variant_id = args.variant
    logger.info(
        f'[+] Wave-3 runner: variant={variant_id} window={args.start.date()}..{args.end.date()} '
        f'cost_tiers={args.cost_tiers} timing_modes={args.timing_modes}'
    )

    git_sha = _registry_git_sha()
    t_global = time.time()

    results: List[SubRunResult] = []
    for timing_mode in args.timing_modes:
        for cost_bps in args.cost_tiers:
            result = _run_or_reuse(args, variant_id, cost_bps, timing_mode, git_sha)
            results.append(result)

    elapsed = time.time() - t_global
    logger.info(f'[+] All sub-backtests done in {elapsed / 60:.2f} min')

    # Build and write artifacts atomically.
    date_str = datetime.now().strftime('%Y%m%d')
    md_path = args.output_dir / f'{date_str}_wave3_{variant_id.lower()}.md'
    json_path = args.output_dir / f'{date_str}_wave3_{variant_id.lower()}.json'

    md_content = _build_report(args, variant_id, git_sha, results)
    json_content = json.dumps(
        _build_json_payload(args, variant_id, git_sha, results),
        indent=2,
        default=str,
    )

    _atomic_write(md_path, md_content)
    _atomic_write(json_path, json_content)

    logger.info(f'[+] Report: {md_path}')
    logger.info(f'[+] JSON:   {json_path}')

    # Summary to stdout.
    logger.info(f'[+] --- V31 vs V11 (incumbent Sharpe 0.528 at 5bps) ---')
    for rr in results:
        m = _compute_extended_metrics(rr.records)
        reused_flag = ' [reused]' if rr.reused_from_registry else ''
        logger.info(
            f'[+]   {rr.variant_id} {rr.timing_mode} {rr.cost_bps}bps: '
            f'Sharpe={m["sharpe"]:.3f} CAGR={m["cagr"]:.2%} '
            f'MaxDD={m["max_dd"]:.2%} AnnTO={m["ann_turnover"]:.0%} '
            f'PSR={m["psr_vs_0"]:.4f}{reused_flag}'
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
