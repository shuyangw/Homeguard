"""Redownload Alpaca US-equity 1-min bars at SIP feed (raw + split-adjusted).

Usage:
    python scripts/data/redownload_sip_equities.py \
        --threads 12 --feeds raw,split

Two-pass orchestrator:
    1. equities_1min_sip_raw   (DataFeed.SIP, Adjustment.RAW)
    2. equities_1min_sip_split (DataFeed.SIP, Adjustment.SPLIT)

Resume semantics: every invocation reaps any in-progress entries to pending
and skips symbols already marked complete. Use --retry-failed to also re-queue
symbols with status=failed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import logging

from src.settings import get_local_storage_dir, get_output_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


FEEDS_AVAILABLE = ("raw", "split")
FEED_TO_SUBDIR = {
    "raw": "equities_1min_sip_raw",
    "split": "equities_1min_sip_split",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Redownload Alpaca US-equity 1-min bars at SIP feed.",
    )
    parser.add_argument(
        "--threads", type=int, default=12,
        help="Worker threads per pass (default 12)",
    )
    parser.add_argument(
        "--start", type=str, default="2015-01-01",
        help="Earliest date to probe (default 2015-01-01)",
    )
    parser.add_argument(
        "--end", type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date (default: today UTC)",
    )
    parser.add_argument(
        "--symbols-from", type=str, default="alpaca",
        help="'alpaca' to snapshot live, or path to CSV with Symbol column",
    )
    parser.add_argument(
        "--feeds", type=str, default="raw,split",
        help="Comma-separated subset of {raw,split} (default both)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-queue symbols with status=failed in the manifest",
    )
    return parser.parse_args(argv)


def setup_file_logger(log_path: Path) -> None:
    """Tee root logger output into a per-run log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    )
    logging.getLogger().addHandler(file_handler)


def main(argv=None) -> int:
    args = parse_args(argv)

    requested = [f.strip() for f in args.feeds.split(",") if f.strip()]
    invalid = [f for f in requested if f not in FEEDS_AVAILABLE]
    if invalid:
        logger.error(f"Invalid feeds: {invalid}. Must be subset of {FEEDS_AVAILABLE}")
        return 2

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_dir = get_output_dir() / "data_acquisition"
    log_path = log_dir / f"sip_redownload-{ts}.log"
    setup_file_logger(log_path)

    logger.info(f"=== SIP redownload starting (log={log_path}) ===")
    logger.info(
        f"Args: threads={args.threads}, start={args.start}, end={args.end}, "
        f"feeds={requested}, retry_failed={args.retry_failed}, "
        f"symbols_from={args.symbols_from}"
    )
    logger.info(f"Storage base: {get_local_storage_dir()}")
    logger.info("(Universe resolution and download passes not yet implemented.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
