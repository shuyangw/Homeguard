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

import pandas as pd
from alpaca.trading.client import TradingClient

from src.api_key import API_KEY, API_SECRET
from src.data.acquisition.alpaca_universe import list_active_us_equities
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


def resolve_universe(
    symbols_from: str, universe_dir: Path
) -> list[str]:
    """Resolve universe from --symbols-from arg.

    'alpaca' -> live API snapshot to dated CSV in config/universes/
    <path>   -> load from existing CSV (must have Symbol column)
    """
    if symbols_from.lower() == "alpaca":
        date_str = datetime.utcnow().strftime("%Y%m%d")
        save_to = universe_dir / f"alpaca_active-{date_str}.csv"
        client = TradingClient(API_KEY, API_SECRET, paper=False)
        return list_active_us_equities(client, save_to=save_to)

    csv_path = Path(symbols_from)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    col = next(
        (c for c in df.columns if c.lower() in ("symbol", "ticker")),
        None,
    )
    if col is None:
        raise ValueError(f"No Symbol/Ticker column in {csv_path}")
    symbols = (
        df[col].dropna().astype(str).str.strip().str.upper().tolist()
    )
    return sorted(set(s for s in symbols if s))


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

    universe_dir = PROJECT_ROOT / "config" / "universes"
    try:
        universe = resolve_universe(args.symbols_from, universe_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Universe resolution failed: {e}")
        return 3
    logger.info(f"Universe size: {len(universe)} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
