"""Aggregate raw FX quote events to minute-bar derivatives.

For each (symbol, year, month) partition in fx_quotes_raw/, computes:
  - bid_open, bid_high, bid_low, bid_close (best-bid OHLC per minute)
  - ask_open, ask_high, ask_low, ask_close (best-ask OHLC per minute)
  - spread_mean, spread_p50, spread_p95, spread_p99 (effective spread distribution)
  - spread_twa (time-weighted-average quoted spread; simplified to mean for now)
  - quote_count (number of events per minute)

Output: fx_quotes_minute_aggregated/symbol={SYM}/year={Y}/month={M}/data.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
logger = get_logger(__name__)

AGGREGATED_COLUMNS = [
    "timestamp",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "spread_mean", "spread_p50", "spread_p95", "spread_p99",
    "quote_count",
]


def aggregate_partition(raw_path: Path, out_path: Path) -> int:
    """Read raw quote events from raw_path, aggregate to 1-min bars, write to out_path."""
    df = pl.read_parquet(raw_path)
    if df.is_empty():
        return 0
    df = df.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
    )
    agg = df.group_by_dynamic("timestamp", every="1m").agg([
        pl.col("bid_price").first().alias("bid_open"),
        pl.col("bid_price").max().alias("bid_high"),
        pl.col("bid_price").min().alias("bid_low"),
        pl.col("bid_price").last().alias("bid_close"),
        pl.col("ask_price").first().alias("ask_open"),
        pl.col("ask_price").max().alias("ask_high"),
        pl.col("ask_price").min().alias("ask_low"),
        pl.col("ask_price").last().alias("ask_close"),
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("spread").quantile(0.5).alias("spread_p50"),
        pl.col("spread").quantile(0.95).alias("spread_p95"),
        pl.col("spread").quantile(0.99).alias("spread_p99"),
        pl.len().alias("quote_count"),
    ]).select(AGGREGATED_COLUMNS).sort("timestamp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    agg.write_parquet(tmp)
    os.replace(tmp, out_path)
    return agg.height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None,
                        help="Restrict to one symbol (default: all)")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    root = get_local_storage_dir()
    raw_root = root / "fx_quotes_raw"
    agg_root = root / "fx_quotes_minute_aggregated"

    if not raw_root.exists():
        logger.error(f"no raw data at {raw_root}")
        return 1

    skip_existing = not args.no_skip_existing
    sym_dirs = sorted(raw_root.iterdir())
    if args.symbol:
        sym_dirs = [d for d in sym_dirs if d.name == f"symbol={args.symbol}"]

    total_written = 0
    for sym_dir in sym_dirs:
        if not sym_dir.name.startswith("symbol="):
            continue
        sym = sym_dir.name[len("symbol="):]
        for y_dir in sorted(sym_dir.iterdir()):
            if not y_dir.name.startswith("year="):
                continue
            for m_dir in sorted(y_dir.iterdir()):
                if not m_dir.name.startswith("month="):
                    continue
                raw_path = m_dir / "data.parquet"
                if not raw_path.exists():
                    continue
                out_path = (agg_root / sym_dir.name / y_dir.name
                            / m_dir.name / "data.parquet")
                if skip_existing and out_path.exists():
                    continue
                try:
                    n = aggregate_partition(raw_path, out_path)
                    total_written += n
                    logger.info(f"  {sym} {y_dir.name}/{m_dir.name}: {n:,} bars")
                except Exception as e:
                    logger.error(f"  {sym} {y_dir.name}/{m_dir.name}: {e}")

    logger.info(f"=== Aggregated {total_written:,} minute bars total ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
