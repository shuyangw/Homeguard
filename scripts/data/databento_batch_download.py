"""Poll Databento batch jobs to completion, download files, convert to parquet.

Reads job IDs from output/databento_batch_jobs.json (written by
databento_batch_submit.py). Polls every 30s until all jobs are done, then
downloads each job's files to H:/Stock_Data/futures_dbn_staging/<section>/
and converts them to canonical parquet under H:/Stock_Data/futures_*/.

Idempotent: re-running picks up where it left off (downloads cached, parquets
not re-written if present).

Usage:
    python scripts/data/databento_batch_download.py
    python scripts/data/databento_batch_download.py --skip-convert  # download only
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import databento as db

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATE_FILE = PROJECT_ROOT / "output" / "databento_batch_jobs.json"
STAGING_ROOT = get_local_storage_dir() / "futures_dbn_staging"
POLL_INTERVAL = 30  # seconds

# Per-section storage roots and conversion behavior
SECTION_CONFIG = {
    "A_v": {
        "subdir": "futures_1min",
        "schema": "ohlcv-1m",
        "partition": "symbol_year_month",  # symbol=X/year=Y/month=M/data.parquet
    },
    "A_n_diag": {
        "subdir": "futures_1min_oi_roll",
        "schema": "ohlcv-1m",
        "partition": "symbol_year_month",
    },
    "B": {
        "subdir": "futures_per_contract",
        "schema": "ohlcv-1d",
        "partition": "root_year",  # root=X/year=Y/data.parquet
    },
    "C": {
        "subdir": "futures_options",
        "schema": "ohlcv-1d",
        "partition": "root_year",
    },
    "D": {
        "subdir": "futures_definitions",
        "schema": "definition",
        "partition": "year_month",  # year=Y/month=M/data.parquet (all symbols mixed)
    },
    "E": {
        "subdir": "futures_statistics",
        "schema": "statistics",
        "partition": "year_month",
    },
    "F": {
        "subdir": "futures_mbp1",
        "schema": "mbp-1",
        "partition": "symbol_year_month_day",
    },
}


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State file missing: {STATE_FILE}. Run databento_batch_submit.py first."
        )
    return json.loads(STATE_FILE.read_text())


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def poll_until_all_done(client: db.Historical, state: dict[str, Any]) -> None:
    """Poll job listings every POLL_INTERVAL until every section reaches done state."""
    target_ids = {info["id"]: section for section, info in state["jobs"].items()}
    pending = set(target_ids.keys())
    t0 = time.time()
    last_status: dict[str, str] = {}
    last_heartbeat = 0.0

    while pending:
        try:
            jobs = client.batch.list_jobs(states=["queued", "processing", "done"])
        except Exception as e:
            logger.warning(f"  list_jobs failed: {e}; retrying after {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        seen_now = set()
        status_now = {}
        for j in jobs:
            jid = j.get("id")
            if jid in target_ids:
                seen_now.add(jid)
                status_now[jid] = j.get("state", "unknown")
                # Persist current state into state file
                section = target_ids[jid]
                state["jobs"][section]["state"] = j.get("state")
                state["jobs"][section]["record_count"] = j.get("record_count")
                state["jobs"][section]["actual_size"] = j.get("actual_size")
                state["jobs"][section]["package_size"] = j.get("package_size")
                if j.get("ts_process_done"):
                    state["jobs"][section]["ts_process_done"] = j["ts_process_done"]

        # Anything we lost from the listing means it transitioned to "done"
        # and dropped off (list_jobs returns active/done only).
        for jid in pending - seen_now:
            section = target_ids[jid]
            state["jobs"][section]["state"] = "done"
            status_now[jid] = "done"

        # Print status only when something changes
        if status_now != last_status:
            logger.info(f"[{time.time() - t0:.0f}s] Status:")
            for section, info in state["jobs"].items():
                rc = info.get("record_count")
                rc_str = f"{rc:>15,}" if rc is not None else " " * 15
                size = info.get("actual_size")
                size_str = f"{size:,}" if size is not None else "-"
                logger.info(
                    f"  {section}: {info['state']:>11} records={rc_str} "
                    f"size={size_str}"
                )
            last_status = dict(status_now)
            _save_state(state)
        elif time.time() - last_heartbeat > 60:
            still_processing = ",".join(
                target_ids[jid] for jid in pending
            )
            logger.info(
                f"[heartbeat {time.time()-t0:.0f}s] still processing: {still_processing}"
            )
            last_heartbeat = time.time()

        # Update pending set
        pending = {
            jid
            for jid in pending
            if state["jobs"][target_ids[jid]]["state"] != "done"
        }

        if not pending:
            break
        time.sleep(POLL_INTERVAL)

    logger.info(f"All jobs done. Total wait: {time.time() - t0:.0f}s")
    _save_state(state)


def download_all(client: db.Historical, state: dict[str, Any]) -> None:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    for section, info in state["jobs"].items():
        if info.get("state") != "done":
            logger.info(f"  {section}: state={info.get('state')}, not ready; skipping")
            continue
        section_dir = STAGING_ROOT / section
        section_dir.mkdir(exist_ok=True)
        manifest_marker = section_dir / ".download_complete"
        if manifest_marker.exists():
            logger.info(f"  {section}: already downloaded, skipping")
            continue
        logger.info(f"Downloading {section} (id={info['id']})...")
        t0 = time.time()
        try:
            paths = client.batch.download(job_id=info["id"], output_dir=str(section_dir))
            elapsed = time.time() - t0
            logger.info(f"  {section}: {len(paths)} files in {elapsed:.0f}s")
            manifest_marker.write_text(json.dumps({"file_count": len(paths)}))
        except Exception as e:
            logger.error(f"  {section}: download FAILED {type(e).__name__}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--skip-poll",
        action="store_true",
        help="Skip polling; assume all jobs are already done",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; only poll and exit",
    )
    args = parser.parse_args()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set")
        return 1
    client = db.Historical(api_key)

    state = _load_state()
    logger.info(f"Loaded {len(state['jobs'])} jobs from state file")

    if not args.skip_poll:
        poll_until_all_done(client, state)

    if not args.skip_download:
        download_all(client, state)

    logger.info("\nDone. Conversion to parquet runs separately:")
    logger.info(f"  Staging: {STAGING_ROOT}")
    logger.info("  Run: python scripts/data/databento_batch_convert.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
