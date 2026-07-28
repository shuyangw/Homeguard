#!/usr/bin/env python3
"""Materialize regime_state_daily by replaying the production regime detector
against the MATERIALIZED VIX spot snapshot + SPY daily bars.

Usage:
    python scripts/data/build_regime_state_daily.py [--start 2012-06-01] [--end 2026-02-28]
"""
from __future__ import annotations

import argparse
import sys

from src.data.acquisition.plugins.vix_spot import REQUIRED_START
from src.data.derivations.regime_state import build
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize regime_state_daily.")
    parser.add_argument("--start", default=REQUIRED_START)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    with RunStatus("build_regime_state_daily",
                   meta={"start": args.start, "end": args.end}) as status:
        path = build(start=args.start, end=args.end, status=status)
    logger.info(f"[+] regime_state_daily materialized -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
