"""Uploads the console status document to S3: python -m src.console_agent.upload --reason periodic|shutdown.

The document is the agent's own /status shape plus the full latest decision per
strategy, so the local console renders one shape whichever source it read.
Uses the AWS CLI already on the instance, so the venv needs no boto3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from dotenv import load_dotenv

from src.console_agent import status
from src.settings import get_local_storage_dir
from src.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
BUCKET_ENV = "CONSOLE_SNAPSHOT_BUCKET"
OBJECT_KEY = "console/latest/status.json"
# Fits inside the shutdown unit's TimeoutStopSec=60 with room for the CLI's own retries.
UPLOAD_TIMEOUT_SECONDS = 45
REASONS = ("periodic", "shutdown")


def build_document(config: status.AgentConfig, now: datetime, reason: str) -> dict:
    document = status.build_status(config, now)
    decisions = {}
    for name in document["strategies"]:
        try:
            decisions[name] = status.read_latest_decision(config.latest_dir, name)
        except status.READ_ERRORS:
            # build_status already put this read failure in document["errors"].
            decisions[name] = None
    return {"uploaded_at": now.isoformat(), "reason": reason, "status": document, "decisions": decisions}


def upload(document: dict, bucket: str, run: Callable = subprocess.run) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(document, handle, default=str)
        tmp_path = Path(handle.name)
    try:
        run(
            ["aws", "s3", "cp", str(tmp_path), f"s3://{bucket}/{OBJECT_KEY}", "--only-show-errors",
             # Short CLI timeouts make it retry and name the unreachable endpoint before we kill it.
             "--cli-connect-timeout", "5", "--cli-read-timeout", "10"],
            capture_output=True, text=True, timeout=UPLOAD_TIMEOUT_SECONDS, check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[console-upload] aws s3 cp failed with exit {e.returncode}: {(e.stderr or '').strip()}")
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"[console-upload] aws s3 cp did not complete: {e!r}")
    finally:
        tmp_path.unlink(missing_ok=True)
    return False


def run_upload(config: status.AgentConfig, environ: Mapping[str, str], reason: str, now: datetime,
               run: Callable = subprocess.run) -> int:
    bucket = environ.get(BUCKET_ENV, "").strip()
    if not bucket:
        logger.error(f"[console-upload] {BUCKET_ENV} is not set")
        return 1
    if not upload(build_document(config, now, reason), bucket, run):
        return 1
    logger.info(f"[console-upload] uploaded {reason} snapshot to s3://{bucket}/{OBJECT_KEY}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the Homeguard Console status document to S3")
    parser.add_argument("--reason", choices=REASONS, required=True)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    snapshot_dir = Path(get_local_storage_dir()) / "metrics_snapshots"
    config = status.AgentConfig(repo_root=REPO_ROOT, snapshot_dir=snapshot_dir, operator_login="")
    raise SystemExit(run_upload(config, os.environ, args.reason, datetime.now(timezone.utc)))


if __name__ == "__main__":
    main()
