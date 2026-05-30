"""Markdown report builders for Phase B harness output."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
import pandas as pd

from src.research.regime_momentum_lab.metrics import (
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


def _row(label, v01_val, v03_val, fmt='{:.2%}'):
    delta = v03_val - v01_val
    return f'| {label} | {fmt.format(v01_val)} | {fmt.format(v03_val)} | {fmt.format(delta)} |'


def build_parity_report(*, v01_records: List, v03_records: List, cost_bps: float) -> str:
    eq01, ret01 = _equity_curve(v01_records), _returns(v01_records)
    eq03, ret03 = _equity_curve(v03_records), _returns(v03_records)

    out: List[str] = []
    out.append('# Phase 4 V01 vs V03 Parity Finding\n')
    out.append('## Question\n')
    out.append('Does applying crash exposure correctly (V03) improve net Sharpe over the')
    out.append('fresh-portfolio baseline (V01) that ignores crash exposure?\n')
    out.append(f'## Side-by-side at {cost_bps} bps per side\n')
    out.append('| Metric | V01 | V03 | Delta (V03 - V01) |')
    out.append('|---|---:|---:|---:|')
    out.append(_row('Sharpe',         sharpe_ratio(ret01), sharpe_ratio(ret03), fmt='{:.3f}'))
    out.append(_row('CAGR',           cagr(eq01),           cagr(eq03)))
    out.append(_row('Max DD',         max_drawdown(eq01),   max_drawdown(eq03)))
    out.append(_row('Avg turnover',   avg_daily_turnover(v01_records), avg_daily_turnover(v03_records)))
    out.append(_row('Cost drag',      cost_drag_pct(v01_records),      cost_drag_pct(v03_records)))
    out.append('')
    out.append('## Conclusion\n')
    out.append('Pick ONE based on the metrics:\n')
    out.append('1. **V03 wins net.** Advance to Wave 1 turnover-control on V03 base.')
    out.append('2. **V03 wins gross but loses net to cost.** Phase 3A generalized; turnover-control needed before V03 is viable.')
    out.append('3. **No material difference.** Investigate signal/regime overlay/sector concentration in Phase C.\n')
    out.append('## Next step\n')
    out.append('Documented in docs/progress/<this-session>.md at completion.')
    return '\n'.join(out)
