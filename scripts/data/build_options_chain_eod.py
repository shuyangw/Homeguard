"""CLI entry point for materializing the options_chain_eod table.

Usage:
    python scripts/data/build_options_chain_eod.py --root SPY --year 2024 --month 1
    python scripts/data/build_options_chain_eod.py --root SPY            # all months

Writes <storage>/options/options_chain_eod/root=X/year=YYYY/month=MM/data.parquet
and a RunStatus heartbeat under output/run_status/.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.options.canonical import main

if __name__ == "__main__":
    raise SystemExit(main())
