#!/usr/bin/env python3
"""Run a regime-momentum backtest variant against SIP data and emit a Markdown report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.research.regime_momentum_lab.config import HarnessConfig
from src.research.regime_momentum_lab.engine import run_variant
from src.research.regime_momentum_lab.variants import REGISTRY
from src.research.regime_momentum_lab.reports import build_variant_report


def _git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def _format_turnover_line(variant: str, bps: float, records=None, _turnover: float | None = None) -> str:
    """One-line realized-turnover summary. Pass _turnover directly in tests; in
    production pass `records` and let it route through the metric."""
    if _turnover is None:
        from src.research.regime_momentum_lab.metrics import avg_daily_turnover
        _turnover = avg_daily_turnover(records)
    return (f'[turnover] {variant} @ {bps}bps: '
            f'avg_daily_turnover={_turnover:.4f} ({_turnover * 100:.2f}% of portfolio/day)')


def _validate_rebalance_frequency(freq: str) -> None:
    """Fail loud on cadences the engine does not yet honor (PR 6 builds weekly)."""
    if freq != 'daily':
        raise NotImplementedError(
            f'rebalance_frequency={freq!r} is not implemented in the engine yet '
            f'(see PR 6, gated on the daily cost verdict). Only "daily" is supported. '
            f'The HarnessConfig field exists but engine.run_variant does not branch on it.')


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run a regime-momentum backtest variant.')
    p.add_argument('--variant', required=True, choices=list(REGISTRY.keys()))
    p.add_argument('--start', required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    p.add_argument('--end', required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    p.add_argument('--cost-bps', type=str, default='0,2.5,5,7.5',
                   help='Comma-separated cost tiers in bps per side.')
    p.add_argument('--timing', choices=['near_close', 'one_day_lag'], default='near_close')
    p.add_argument('--universe', type=Path,
                   default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rebalance-frequency',
                   choices=['daily', 'weekly_friday', 'weekly_wednesday'],
                   default='daily',
                   help='Rebalance cadence. Only "daily" is implemented today; '
                        'weekly_* is built in PR 6 (gated on the daily cost verdict).')
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    _validate_rebalance_frequency(args.rebalance_frequency)
    spec = REGISTRY[args.variant]
    tiers = [float(t) for t in args.cost_bps.split(',') if t.strip()]

    records_by_tier = {}
    for bps in tiers:
        cfg = HarnessConfig(
            start_date=args.start,
            end_date=args.end,
            universe_csv=args.universe,
            initial_capital=args.initial_capital,
            cost_bps_per_side=bps,
            timing_mode=args.timing,
            rebalance_frequency=args.rebalance_frequency,
        )
        from src.utils.logger import logger
        logger.info(f'[phase4] Running {args.variant} at {bps} bps...')
        records = run_variant(cfg, spec)
        records_by_tier[bps] = records

    md = build_variant_report(
        variant_id=args.variant,
        variant_description=spec.description,
        records_by_cost_bps=records_by_tier,
        git_commit=_git_sha(),
        universe_csv=str(args.universe),
        timing_mode=args.timing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f'wrote {args.output}')
    for bps, records in records_by_tier.items():
        print(_format_turnover_line(args.variant, bps, records=records))
    return 0


if __name__ == '__main__':
    sys.exit(main())
