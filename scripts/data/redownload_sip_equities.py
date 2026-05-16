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
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.trading.client import TradingClient

from src.api_key import API_KEY, API_SECRET
from src.data.acquisition.alpaca_universe import list_active_us_equities
from src.data.acquisition.plugins.alpaca_equities import AlpacaEquitiesPlugin
from src.data.acquisition.status_tracker import rebuild_tracker, write_tracker_csv
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
        client = TradingClient(API_KEY, API_SECRET, paper=True)
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


PASS_CONFIGS = {
    "raw": (DataFeed.SIP, Adjustment.RAW, "equities_1min_sip_raw"),
    "split": (DataFeed.SIP, Adjustment.SPLIT, "equities_1min_sip_split"),
}


def run_pass(
    feed_name: str,
    universe: list[str],
    start: str,
    end: str,
    num_threads: int,
    retry_failed: bool,
) -> dict:
    """Run one download pass; returns summary dict."""
    data_feed, adjustment, subdir = PASS_CONFIGS[feed_name]
    base_dir = get_local_storage_dir()

    logger.info(
        f"=== Pass '{feed_name}' starting: feed={data_feed.value} "
        f"adjustment={adjustment.value} subdir={subdir} ==="
    )
    plugin = AlpacaEquitiesPlugin(
        feed=data_feed,
        adjustment=adjustment,
        storage_subdir_override=subdir,
        num_threads=num_threads,
    )

    reaped = plugin.manifest.reap_in_progress()
    if reaped:
        logger.info(f"Reaped {len(reaped)} in-progress entries to pending: {reaped[:10]}...")
        plugin._emit_event("interrupted_reaped", subdir=subdir, count=len(reaped))

    completed_symbols: set[str] = {
        sym for sym, e in plugin.manifest.get_all_entries().items()
        if e.get("status") == "complete"
    }
    failed_symbols: set[str] = {
        sym for sym, e in plugin.manifest.get_all_entries().items()
        if e.get("status") == "failed"
    }

    universe_set = set(universe)
    symbols_remaining = sorted(universe_set - completed_symbols)
    if retry_failed:
        symbols_remaining = sorted(set(symbols_remaining) | failed_symbols)

    logger.info(
        f"[{feed_name}] {len(completed_symbols)} complete, "
        f"{len(failed_symbols)} failed, {len(symbols_remaining)} to download"
    )

    result = plugin.download(symbols_remaining, start_date=start, end_date=end)

    tracker_rows = rebuild_tracker(
        base_dir=base_dir, subdir=subdir, universe=universe
    )
    tracker_path = base_dir / "_manifests" / f"{subdir}.status.csv"
    write_tracker_csv(tracker_path, tracker_rows)

    return {
        "feed_name": feed_name,
        "subdir": subdir,
        "result": result,
        "tracker_path": str(tracker_path),
    }


def write_session_summary(
    summary_path: Path,
    args: argparse.Namespace,
    universe_size: int,
    summaries: list[dict],
    total_elapsed_sec: float,
) -> None:
    """Write final session summary markdown."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# SIP Redownload Session Summary",
        f"",
        f"**Started**: {datetime.utcnow().isoformat()}Z",
        f"**Universe size**: {universe_size}",
        f"**Args**: threads={args.threads}, start={args.start}, end={args.end}, "
        f"feeds={args.feeds}, retry_failed={args.retry_failed}",
        f"**Total elapsed**: {total_elapsed_sec:.1f}s",
        f"",
        f"## Per-pass results",
        f"",
    ]
    for s in summaries:
        r = s["result"]
        lines.extend([
            f"### {s['feed_name']} (`{s['subdir']}`)",
            f"",
            f"- succeeded: {r.succeeded}",
            f"- failed:    {r.failed}",
            f"- total_rows: {r.total_rows:,}",
            f"- elapsed:   {r.elapsed_seconds:.1f}s",
            f"- tracker:   `{s['tracker_path']}`",
            f"",
        ])
        if r.failed_symbols:
            lines.append("**Failed symbols:**")
            lines.append("")
            for sym, err in r.failed_symbols[:50]:
                lines.append(f"- `{sym}`: {err}")
            if len(r.failed_symbols) > 50:
                lines.append(f"- ... and {len(r.failed_symbols) - 50} more")
            lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Session summary written: {summary_path}")


def main(argv=None) -> int:
    import time as _time
    run_start = _time.time()

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

    summaries = []
    for feed_name in requested:
        try:
            summary = run_pass(
                feed_name=feed_name,
                universe=universe,
                start=args.start,
                end=args.end,
                num_threads=args.threads,
                retry_failed=args.retry_failed,
            )
            summaries.append(summary)
        except KeyboardInterrupt:
            logger.warning(f"Pass '{feed_name}' interrupted by user")
            raise

    for s in summaries:
        r = s["result"]
        logger.info(
            f"[{s['feed_name']}] succeeded={r.succeeded} failed={r.failed} "
            f"rows={r.total_rows:,} elapsed={r.elapsed_seconds:.1f}s "
            f"tracker={s['tracker_path']}"
        )

    total_elapsed = _time.time() - run_start
    summary_path = (
        get_output_dir() / "data_acquisition"
        / f"sip_redownload-{ts}-summary.md"
    )
    write_session_summary(
        summary_path, args, len(universe), summaries, total_elapsed
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
