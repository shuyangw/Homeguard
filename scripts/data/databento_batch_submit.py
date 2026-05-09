"""Submit Databento batch jobs for all bulk-pull sections.

Replaces the synchronous streaming approach (too slow for full history).
Each section becomes one or more batch jobs which Databento processes
server-side and returns as compressed dbn.zst files for fast download.

Usage:
    python scripts/data/databento_batch_submit.py
        # submits all unsubmitted jobs, records IDs to state file

State file: output/databento_batch_jobs.json (job IDs + metadata).
Pair with databento_batch_download.py to poll, download, and convert.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import databento as db

from src.data.acquisition.plugins.databento_futures import (
    BULK_PULL_UNIVERSE_V,
    BULK_PULL_START,
    DATASET,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_END = "2026-02-22"
MBP1_FREE_START = "2025-08-22"
MBP1_FREE_END = "2026-02-22"

CONTINUOUS_V_DOTTED = [f"{s}.v.0" for s in BULK_PULL_UNIVERSE_V]
CONTINUOUS_DIAGNOSTIC = ["GC.n.0"]

ALL_FUT_PARENTS = [f"{s}.FUT" for s in BULK_PULL_UNIVERSE_V]
ALL_OPT_PARENTS = [
    "ES.OPT", "NQ.OPT", "RTY.OPT",
    "CL.OPT", "NG.OPT",
    "GC.OPT", "SI.OPT",
    "ZN.OPT", "ZB.OPT",
    "6E.OPT", "6J.OPT",
    "ZC.OPT", "ZS.OPT",
]

MBP1_SYMBOLS = ["ES.v.0", "MES.v.0", "NQ.v.0", "MNQ.v.0"]

STATE_FILE = PROJECT_ROOT / "output" / "databento_batch_jobs.json"


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"jobs": {}}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _submit(
    client: db.Historical,
    section: str,
    schema: str,
    symbols: list[str],
    stype_in: str,
    start: str,
    end: str,
    split_duration: str | None,
    split_symbols: bool,
    state: dict[str, Any],
) -> None:
    """Submit one batch job and record its id in state."""
    if section in state["jobs"]:
        logger.info(f"  Section {section}: already submitted "
                    f"(id={state['jobs'][section]['id']}), skipping")
        return
    logger.info(f"Submitting Section {section}: {schema}, {len(symbols)} symbols, "
                f"{start} -> {end}, split_duration={split_duration}, "
                f"split_symbols={split_symbols}")

    kwargs = dict(
        dataset=DATASET,
        schema=schema,
        symbols=symbols,
        stype_in=stype_in,
        start=start,
        end=end,
        encoding="dbn",
        compression="zstd",
        split_symbols=split_symbols,
        delivery="download",
    )
    if split_duration:
        kwargs["split_duration"] = split_duration

    try:
        job = client.batch.submit_job(**kwargs)
    except Exception as e:
        logger.error(f"  Section {section}: SUBMISSION FAILED {type(e).__name__}: {e}")
        return

    state["jobs"][section] = {
        "id": job["id"],
        "schema": schema,
        "symbols_count": len(symbols),
        "stype_in": stype_in,
        "start": start,
        "end": end,
        "split_duration": split_duration,
        "split_symbols": split_symbols,
        "state": job["state"],
        "cost_usd": job["cost_usd"],
        "ts_received": job["ts_received"],
    }
    _save_state(state)
    logger.info(f"  Section {section}: id={job['id']} state={job['state']} "
                f"cost=${job['cost_usd']}")


def submit_all(end_date: str = DEFAULT_END) -> int:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set")
        return 1
    client = db.Historical(api_key)
    state = _load_state()

    # Section A: continuous OHLCV-1m, .v.0 universe
    # 54 symbols x 192 months = 10,368 files with split_symbols=True; under 100k limit
    _submit(
        client, "A_v",
        schema="ohlcv-1m",
        symbols=CONTINUOUS_V_DOTTED,
        stype_in="continuous",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=True,
        state=state,
    )

    # Section A diagnostic: GC.n.0 (open-interest roll)
    _submit(
        client, "A_n_diag",
        schema="ohlcv-1m",
        symbols=CONTINUOUS_DIAGNOSTIC,
        stype_in="continuous",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=True,
        state=state,
    )

    # Section B: per-contract daily OHLCV-1d
    # split_symbols=False because parent symbology expands per contract
    # (each .FUT has hundreds of unique contracts) which blows past 100k file limit.
    # Will split client-side post-download.
    _submit(
        client, "B",
        schema="ohlcv-1d",
        symbols=ALL_FUT_PARENTS,
        stype_in="parent",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=False,
        state=state,
    )

    # Section C: options on futures daily
    # split_symbols=False (option chains have thousands of strike-expiries per family).
    _submit(
        client, "C",
        schema="ohlcv-1d",
        symbols=ALL_OPT_PARENTS,
        stype_in="parent",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=False,
        state=state,
    )

    # Section D already submitted manually; keep its id if known
    # (Will be submitted here if state file is fresh)
    _submit(
        client, "D",
        schema="definition",
        symbols=ALL_FUT_PARENTS + ALL_OPT_PARENTS,
        stype_in="parent",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=False,  # keep one file per month, all symbols
        state=state,
    )

    # Section E: statistics
    # split_symbols=False (statistics include per-contract stats per stat type;
    # ~thousands of unique contracts per family blows past 100k limit).
    _submit(
        client, "E",
        schema="statistics",
        symbols=ALL_FUT_PARENTS,
        stype_in="parent",
        start=BULK_PULL_START, end=end_date,
        split_duration="month",
        split_symbols=False,
        state=state,
    )

    # Section F: MBP-1 last 6 months free; 4 syms x ~130 days = 520 files
    _submit(
        client, "F",
        schema="mbp-1",
        symbols=MBP1_SYMBOLS,
        stype_in="continuous",
        start=MBP1_FREE_START, end=MBP1_FREE_END,
        split_duration="day",
        split_symbols=True,
        state=state,
    )

    # Section Trades_ES_MES: trades schema for ES, MES, last-hour window, 5 years.
    # For Adaptation A imbalance signal proxy (cheaper than MBP-1).
    # Time window: 19:00-21:00 UTC daily (3pm-4pm ET cash session close).
    # Note: Databento batch jobs don't natively support time-of-day filtering;
    # we pull full days and filter post-download in the converter.
    _submit(
        client, "Trades_ES_MES",
        schema="trades",
        symbols=["ES.v.0", "MES.v.0"],
        stype_in="continuous",
        start="2021-01-01", end="2026-02-22",
        split_duration="month",
        split_symbols=True,
        state=state,
    )

    logger.info(f"\nAll sections submitted. State file: {STATE_FILE}")
    logger.info(f"Total jobs: {len(state['jobs'])}")
    for sec, info in state["jobs"].items():
        logger.info(f"  {sec}: {info['id']} ({info['state']}, "
                    f"${info['cost_usd']})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", default=DEFAULT_END)
    args = parser.parse_args()
    return submit_all(end_date=args.end_date)


if __name__ == "__main__":
    sys.exit(main())
