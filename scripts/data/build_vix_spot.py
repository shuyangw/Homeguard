#!/usr/bin/env python3
"""Materialize CBOE VIX INDEX SPOT daily OHLC to local storage.

Usage:
    python scripts/data/build_vix_spot.py [--start 2011-01-01] [--end 2026-07-27]
"""
from __future__ import annotations

import argparse
import sys

from src.data.acquisition.plugins.vix_spot import FETCH_START, build
from src.utils.logger import get_logger
from src.utils.run_status import RunStatus

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize VIX index spot OHLC.")
    parser.add_argument("--start", default=FETCH_START)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    with RunStatus("build_vix_spot", meta={"start": args.start, "end": args.end}):
        path = build(start=args.start, end=args.end)
    logger.info(f"[+] vix_spot materialized -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
