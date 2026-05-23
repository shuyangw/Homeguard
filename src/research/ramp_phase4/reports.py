"""Markdown report builders for Phase B harness output."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.research.ramp_phase4.metrics import (
    sharpe_ratio, cagr, max_drawdown, avg_daily_turnover, cost_drag_pct, regime_attribution,
)
from src.backtesting.statistics.psr import psr
from src.backtesting.validation.deflated_sharpe import compute_deflated_sharpe


DEFAULT_PERIODS: List[Tuple[str, datetime, datetime]] = [
    ('IS 2017-2021',     datetime(2017, 1, 1),  datetime(2021, 12, 31)),
    ('OOS 2022',         datetime(2022, 1, 1),  datetime(2022, 12, 31)),
    ('OOS 2023',         datetime(2023, 1, 1),  datetime(2023, 12, 31)),
    ('OOS 2024',         datetime(2024, 1, 1),  datetime(2024, 12, 31)),
    ('EXT-OOS 2025-26',  datetime(2025, 1, 1),  datetime(2026, 12, 31)),
]


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


def _filter_records_by_period(records, start: datetime, end: datetime):
    """Return only records whose date falls in [start, end] inclusive."""
    out = []
    for r in records:
        d = r.date
        if isinstance(d, pd.Timestamp):
            d = d.to_pydatetime()
        if start <= d <= end:
            out.append(r)
    return out


def _period_metric_value(metric: str, recs) -> str:
    """Compute a single metric for a per-period sub-window.

    Returns 'n/a' when there are not enough records to compute meaningfully.
    Formats are consistent with the single-table view (percent or ratio).
    """
    if not recs:
        return 'n/a'

    if metric == 'CAGR':
        eq = _equity_curve(recs)
        if len(eq) < 2:
            return 'n/a'
        return f'{cagr(eq):.2%}'
    if metric == 'Sharpe':
        rets = _returns(recs)
        if len(rets.dropna()) < 2:
            return 'n/a'
        return f'{sharpe_ratio(rets):.3f}'
    if metric == 'Max DD':
        eq = _equity_curve(recs)
        if len(eq) < 2:
            return 'n/a'
        return f'{max_drawdown(eq):.2%}'
    if metric == 'Avg turnover':
        return f'{avg_daily_turnover(recs):.2%}'
    if metric == 'Cost drag':
        return f'{cost_drag_pct(recs):.2%}'
    return 'n/a'


def build_period_decomposition_table(records, periods: List[Tuple[str, datetime, datetime]]) -> str:
    """Build a per-period decomposition table for a single cost-tier's records.

    Columns: one per period label plus a trailing 'Full' column that reuses
    the entire records list. Rows are the same five metrics shown in the
    single-table view (CAGR, Sharpe, Max DD, Avg turnover, Cost drag).
    Periods with zero records render as 'n/a' across that column.
    """
    labels = [p[0] for p in periods]
    header = '| Metric | ' + ' | '.join(labels) + ' | Full |'
    sep = '|---' + '|---:' * (len(labels) + 1) + '|'
    metric_rows = ['CAGR', 'Sharpe', 'Max DD', 'Avg turnover', 'Cost drag']

    lines = [header, sep]
    period_records = [_filter_records_by_period(records, s, e) for _, s, e in periods]
    for m in metric_rows:
        cells = [_period_metric_value(m, recs) for recs in period_records]
        cells.append(_period_metric_value(m, records))
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _format_statistical_gates(records, n_trials: int) -> str:
    """Render a PSR + DSR markdown gate table for a single cost tier.

    PSR is computed against SR=0 using Pearson kurtosis (normal=3). DSR uses
    the project's compute_deflated_sharpe wrapper which handles excess
    kurtosis internally and applies the n_trials multi-trial correction.
    """
    rets = np.asarray([r.daily_return for r in records], dtype=np.float64)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 30:
        return '_Insufficient data for statistical gates._'

    std = float(rets.std())
    if std <= 0.0:
        return '_Insufficient data for statistical gates._'

    sr_daily = float(rets.mean()) / std
    sr_annual = sr_daily * float(np.sqrt(252))
    skew = float(np.mean(((rets - rets.mean()) / std) ** 3))
    pearson_kurt = float(np.mean(((rets - rets.mean()) / std) ** 4))

    psr_value = psr(sr_annual, 0.0, len(rets), skew, pearson_kurt)
    dsr = compute_deflated_sharpe(rets, n_trials=n_trials)

    psr_pass = 'PASS' if psr_value > 0.95 else 'FAIL'
    dsr_pass = 'PASS' if dsr.passed else 'FAIL'

    lines = [
        '| Gate | Value | Pass threshold | Result |',
        '|---|---:|---:|:---:|',
        f'| PSR (vs SR=0) | {psr_value:.4f} | > 0.95 | {psr_pass} |',
        f'| DSR (n_trials={n_trials}) | p={dsr.p_value:.4f} | p < 0.05 | {dsr_pass} |',
        f'| Expected max Sharpe (null) | {dsr.expected_max_sharpe:.3f} | informational | - |',
        f'| Observed annualized Sharpe | {dsr.observed_sharpe:.3f} | informational | - |',
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
    n_trials: int = 20,
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
        out.append(f'### {bps} bps per side -- per-period\n')
        out.append(build_period_decomposition_table(records_by_cost_bps[bps], DEFAULT_PERIODS))
        out.append('')

    # Use the 5 bps tier (or the highest available) for regime attribution.
    pivot_bps = 5.0 if 5.0 in records_by_cost_bps else max(records_by_cost_bps.keys())
    out.append(f'## Regime attribution ({pivot_bps} bps tier)\n')
    out.append(_format_regime_attribution(records_by_cost_bps[pivot_bps]))
    out.append('')

    out.append(f'## Statistical gates (5 bps tier, n_trials={n_trials})\n')
    out.append(_format_statistical_gates(records_by_cost_bps[pivot_bps], n_trials=n_trials))
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
