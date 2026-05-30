#!/usr/bin/env python3
"""CLI to run a Phase 4 variant against Alpaca SIP data and emit a Markdown report."""

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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run a Phase 4 variant.')
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
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
