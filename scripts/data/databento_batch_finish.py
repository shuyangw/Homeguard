"""Run unattended to finish the bulk pull: poll until all jobs done, download
remaining sections, convert remaining sections to parquet, exit.

Idempotent. Safe to leave running overnight. Will pick up wherever the previous
runs left off (state file persists; .download_complete markers; existing parquet
output is skipped).

Usage:
    python scripts/data/databento_batch_finish.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
POLL_INTERVAL = 60  # seconds


def _load_state() -> dict:
    return json.loads(STATE_FILE.read_text())


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def poll_until_all_done(client: db.Historical) -> None:
    """Poll list_jobs until every section in state file reaches done."""
    while True:
        state = _load_state()
        target_ids = {info["id"]: section for section, info in state["jobs"].items()}
        try:
            jobs = client.batch.list_jobs(states=["queued", "processing", "done"])
        except Exception as e:
            logger.warning(f"list_jobs failed: {e}; retrying")
            time.sleep(POLL_INTERVAL)
            continue

        for j in jobs:
            jid = j.get("id")
            if jid in target_ids:
                section = target_ids[jid]
                state["jobs"][section]["state"] = j.get("state")
                state["jobs"][section]["record_count"] = j.get("record_count")
                state["jobs"][section]["actual_size"] = j.get("actual_size")
                state["jobs"][section]["ts_process_done"] = j.get("ts_process_done")

        # Anything missing from active listing must be done (and dropped from active list)
        seen_ids = {j.get("id") for j in jobs}
        for jid, section in target_ids.items():
            if jid not in seen_ids:
                state["jobs"][section]["state"] = "done"

        _save_state(state)

        pending = [
            section for section, info in state["jobs"].items()
            if info.get("state") != "done"
        ]
        if not pending:
            logger.info("All jobs reached done state.")
            return

        logger.info(f"Pending: {pending}")
        time.sleep(POLL_INTERVAL)


def download_remaining(client: db.Historical) -> None:
    state = _load_state()
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    for section, info in state["jobs"].items():
        if info.get("state") != "done":
            continue
        section_dir = STAGING_ROOT / section
        section_dir.mkdir(exist_ok=True)
        marker = section_dir / ".download_complete"
        if marker.exists():
            logger.info(f"  {section}: already downloaded")
            continue
        logger.info(f"Downloading {section} (id={info['id']})...")
        t0 = time.time()
        try:
            paths = client.batch.download(
                job_id=info["id"], output_dir=str(section_dir)
            )
            elapsed = time.time() - t0
            logger.info(f"  {section}: {len(paths)} files in {elapsed:.0f}s")
            marker.write_text(json.dumps({"file_count": len(paths)}))
        except Exception as e:
            logger.error(f"  {section}: download failed {type(e).__name__}: {e}")


def convert_remaining() -> None:
    """Run convert script for each section that has staged files but no parquet output."""
    state = _load_state()
    convert_script = PROJECT_ROOT / "scripts" / "data" / "databento_batch_convert.py"
    pybin = sys.executable

    output_dir_for_section = {
        "A_v": "futures_1min",
        "A_n_diag": "futures_1min_oi_roll",
        "B": "futures_per_contract_1min",
        "C": "futures_options_1min",
        "D": "futures_definitions",
        "E": "futures_statistics",
        "F": "futures_mbp1",
    }

    for section in state["jobs"].keys():
        section_dir = STAGING_ROOT / section
        if not (section_dir / ".download_complete").exists():
            logger.info(f"  {section}: not downloaded, skipping convert")
            continue
        # Skip if output already exists
        out_root = get_local_storage_dir() / output_dir_for_section.get(section, "")
        if out_root.exists() and any(out_root.rglob("data.parquet")):
            logger.info(f"  {section}: output exists at {out_root}, skipping")
            continue
        logger.info(f"Converting {section}...")
        t0 = time.time()
        result = subprocess.run(
            [pybin, str(convert_script), "--section", section],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"  {section}: converted in {elapsed:.0f}s")
        else:
            logger.error(f"  {section}: convert failed (rc={result.returncode}): "
                         f"{result.stderr[-500:]}")


def main() -> int:
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set")
        return 1
    client = db.Historical(api_key)

    logger.info("=" * 60)
    logger.info("Stage 1: Poll until all jobs done")
    logger.info("=" * 60)
    poll_until_all_done(client)

    logger.info("=" * 60)
    logger.info("Stage 2: Download any remaining sections")
    logger.info("=" * 60)
    download_remaining(client)

    logger.info("=" * 60)
    logger.info("Stage 3: Convert any remaining sections")
    logger.info("=" * 60)
    convert_remaining()

    logger.info("\n*** All stages complete ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
