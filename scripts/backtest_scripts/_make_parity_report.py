#!/usr/bin/env python3
"""One-shot driver for the V01-vs-V03 parity report.

Re-runs both variants for a single cost-bps tier and emits the parity Markdown
via reports.build_parity_report. Used for Phase B Task 19.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.research.regime_momentum_lab.config import HarnessConfig
from src.research.regime_momentum_lab.engine import run_variant
from src.research.regime_momentum_lab.variants import REGISTRY
from src.research.regime_momentum_lab.reports import build_parity_report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Emit the V01-vs-V03 parity report.')
    p.add_argument('--start', required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    p.add_argument('--end', required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    p.add_argument('--cost-bps', type=float, default=5.0)
    p.add_argument('--timing', choices=['near_close', 'one_day_lag'], default='near_close')
    p.add_argument('--universe', type=Path, default=Path('config/universes/sp500-2025.csv'))
    p.add_argument('--initial-capital', type=float, default=100000.0)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    cfg = HarnessConfig(
        start_date=args.start,
        end_date=args.end,
        universe_csv=args.universe,
        initial_capital=args.initial_capital,
        cost_bps_per_side=args.cost_bps,
        timing_mode=args.timing,
    )
    from src.utils.logger import logger
    logger.info(f'[parity] V01 at {args.cost_bps} bps...')
    v01_records = run_variant(cfg, REGISTRY['V01'])
    logger.info(f'[parity] V03 at {args.cost_bps} bps...')
    v03_records = run_variant(cfg, REGISTRY['V03'])
    md = build_parity_report(
        v01_records=v01_records,
        v03_records=v03_records,
        cost_bps=args.cost_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
