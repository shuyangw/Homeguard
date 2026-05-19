"""Markdown report builders for Phase B harness output."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
import pandas as pd

from src.research.ramp_phase4.metrics import (
    sharpe_ratio, cagr, max_drawdown, avg_daily_turnover, cost_drag_pct, regime_attribution,
)


def _equity_curve(records) -> pd.Series:
    return pd.Series(
        [r.portfolio_value for r in records],
        index=[r.date for r in records],
    )


def _returns(records) -> pd.Series:
    return pd.Series(
        [r.daily_return for r in records],
        index=[r.date for r in records],
    )


def _format_metric_table(records) -> str:
    eq = _equity_curve(records)
    rets = _returns(records)
    lines = [
        '| Metric | Value |',
        '|---|---:|',
        f'| CAGR | {cagr(eq):.2%} |',
        f'| Sharpe | {sharpe_ratio(rets):.3f} |',
        f'| Max DD | {max_drawdown(eq):.2%} |',
        f'| Avg daily turnover | {avg_daily_turnover(records):.2%} |',
        f'| Cost drag | {cost_drag_pct(records):.2%} |',
    ]
    return '\n'.join(lines)


def _format_regime_attribution(records) -> str:
    attr = regime_attribution(records)
    lines = [
        '| Regime | Days | Net return |',
        '|---|---:|---:|',
    ]
    for regime, d in sorted(attr.items()):
        lines.append(f"| {regime} | {d['days']} | {d['net_return']:.2%} |")
    return '\n'.join(lines)


def build_variant_report(
    *,
    variant_id: str,
    variant_description: str,
    records_by_cost_bps: Dict[float, List],
    git_commit: str,
    universe_csv: str,
    timing_mode: str,
) -> str:
    """Build a full per-variant Markdown report.

    records_by_cost_bps maps cost-tier (in bps) to the list of DailyRecords from that run.
    """
    out: List[str] = []
    out.append(f'# Phase 4 {variant_id} - {variant_description}\n')
    out.append('## Header\n')
    out.append(f'- Variant: {variant_id}')
    out.append(f'- Description: {variant_description}')
    out.append(f'- Code commit: {git_commit}')
    out.append(f'- Data snapshot: Alpaca SIP DuckDB Parquet')
    out.append(f'- Timing mode: {timing_mode}')
    out.append(f"- Cost tiers run: {sorted(records_by_cost_bps.keys())}")
    out.append(f'- Universe: {universe_csv}')
    out.append('- Known limitations: survivorship bias, daily close approximation, no point-in-time index membership\n')

    out.append('## Metrics by cost tier\n')
    for bps in sorted(records_by_cost_bps.keys()):
        out.append(f'### {bps} bps per side\n')
        out.append(_format_metric_table(records_by_cost_bps[bps]))
        out.append('')

    # Use the 5 bps tier (or the highest available) for regime attribution.
    pivot_bps = 5.0 if 5.0 in records_by_cost_bps else max(records_by_cost_bps.keys())
    out.append(f'## Regime attribution ({pivot_bps} bps tier)\n')
    out.append(_format_regime_attribution(records_by_cost_bps[pivot_bps]))
    out.append('')

    return '\n'.join(out)
