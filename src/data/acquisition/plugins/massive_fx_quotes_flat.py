"""Massive (Polygon) flat-file FX quote/BBO downloader.

Reads daily CSV.gz quote files from Massive's S3 bucket (one file = ALL pairs'
quote events for one trading day), filters to target tickers, writes per-symbol-
per-month parquet under fx_quotes_raw/.

Mirrors the pattern in massive_fx_flat.py but for the quote schema. Auth via
the same MASSIVE_S3_* env vars.

Schema (Massive quotes_v1 CSV):
  ticker (e.g. C:EUR-USD), participant_timestamp (ns), bid_price, ask_price,
  bid_exchange, ask_exchange

Canonical Parquet output (fx_quotes_raw/):
  timestamp (Datetime[ns, UTC]), bid_price (Float64), ask_price (Float64),
  bid_exchange (Int32), ask_exchange (Int32)

Storage: per-event, ~1-5M rows per pair per day. Parquet compresses 10-20x.
Process pairs sequentially per day to bound RAM.
"""
from __future__ import annotations

import gzip
import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import boto3
import polars as pl
from botocore.client import Config
from botocore.exceptions import ClientError

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUCKET = "flatfiles"
PREFIX = "global_forex/quotes_v1"

QUOTE_CANONICAL_COLUMNS = [
    "timestamp", "bid_price", "ask_price", "bid_exchange", "ask_exchange",
]


@dataclass(frozen=True)
class TargetPair:
    """One pair to acquire (mirrors massive_fx_flat.TargetPair)."""
    hg_symbol: str           # e.g. "EURUSD"
    massive_ticker: str      # e.g. "C:EUR-USD"
    effective_start: date


def _env(name: str) -> str:
    """Read env var; raise if not set."""
    val = os.environ.get(name)
    if val:
        return val
    proj = Path(__file__).resolve().parents[4]
    env_path = proj / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not in environment or .env")


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=_env("MASSIVE_S3_ENDPOINT"),
        aws_access_key_id=_env("MASSIVE_S3_ACCESS_KEY"),
        aws_secret_access_key=_env("MASSIVE_S3_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )


def key_for(d: date) -> str:
    """S3 key for a daily quote file."""
    return f"{PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"


def fetch_day(s3, day: date) -> bytes | None:
    """Download one daily CSV.gz. Returns raw bytes or None if not present."""
    key = key_for(day)
    buf = io.BytesIO()
    try:
        s3.download_fileobj(BUCKET, key, buf)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None
        raise
    return buf.getvalue()


def parse_day(raw_gz: bytes, target_tickers: set[str]) -> dict[str, list[list]]:
    """Decompress + parse one daily file. Returns {massive_ticker: [row_list, ...]}.

    Returns LIST OF LISTS for memory efficiency (vs list of dicts).
    Row format: [timestamp_ns, bid_price, ask_price, bid_exchange, ask_exchange]
    """
    text = gzip.decompress(raw_gz).decode("utf-8")
    out: dict[str, list[list]] = {}
    lines = text.splitlines()
    # Skip header
    for line in lines[1:]:
        # ticker,participant_timestamp,bid_price,ask_price,bid_exchange,ask_exchange
        comma1 = line.find(",")
        if comma1 < 0:
            continue
        ticker = line[:comma1]
        if ticker not in target_tickers:
            continue
        try:
            fields = line.split(",")
            row = [
                int(fields[1]),                          # timestamp ns
                float(fields[2]),                        # bid_price
                float(fields[3]),                        # ask_price
                int(fields[4]) if fields[4] else 0,     # bid_exchange
                int(fields[5]) if fields[5] else 0,     # ask_exchange
            ]
        except (ValueError, IndexError) as e:
            logger.warning(f"skipping malformed quote for {ticker}: {e!r}")
            continue
        out.setdefault(ticker, []).append(row)
    return out


def rows_to_parquet(rows: list[list], out_path: Path) -> int:
    """Write rows (list of [ts_ns, bid_px, ask_px, bid_ex, ask_ex]) to parquet.

    Schema matches QUOTE_CANONICAL_COLUMNS with Datetime[ns, UTC] + Float64 +
    Float64 + Int32 + Int32. Atomic write via tmp -> rename.
    """
    if not rows:
        return 0
    df = pl.DataFrame(
        rows, schema=QUOTE_CANONICAL_COLUMNS, orient="row",
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="ns", time_zone="UTC")),
        pl.col("bid_price").cast(pl.Float64),
        pl.col("ask_price").cast(pl.Float64),
        pl.col("bid_exchange").cast(pl.Int32),
        pl.col("ask_exchange").cast(pl.Int32),
    ).sort("timestamp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, out_path)
    return df.height


def iter_days_by_month(start: date, end: date) -> Iterable[tuple[int, int, list[date]]]:
    """Yield (year, month, [day1, day2, ...]) for every month touched."""
    current_year, current_month = start.year, start.month
    bucket: list[date] = []
    n = (end - start).days + 1
    for i in range(n):
        d = start + timedelta(days=i)
        if (d.year, d.month) != (current_year, current_month):
            yield current_year, current_month, bucket
            current_year, current_month = d.year, d.month
            bucket = []
        bucket.append(d)
    if bucket:
        yield current_year, current_month, bucket


def download_pairs(
    pairs: list[TargetPair],
    start_date: date,
    end_date: date,
    *,
    storage_root: Path | None = None,
    concurrency: int = 4,
    skip_existing: bool = True,
) -> dict:
    """Bulk-download quote events for `pairs` over [start_date, end_date].

    Output: {storage_root}/fx_quotes_raw/symbol={SYM}/year={Y}/month={M}/data.parquet
    Returns summary dict.
    """
    root = storage_root if storage_root is not None else get_local_storage_dir()
    s3 = make_s3_client()
    by_massive = {p.massive_ticker: p for p in pairs}
    target_set = set(by_massive)

    summary = {
        "total_days_attempted": 0,
        "total_days_present": 0,
        "total_days_missing": 0,
        "rows_per_symbol": {p.hg_symbol: 0 for p in pairs},
        "months_written": 0,
        "months_skipped_existing": 0,
    }

    for year, month, days in iter_days_by_month(start_date, end_date):
        # Per-pair accumulator for this month. List of rows.
        month_buffers: dict[str, list[list]] = defaultdict(list)

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_day = {ex.submit(fetch_day, s3, d): d for d in days}
            for fut in as_completed(future_to_day):
                d = future_to_day[fut]
                summary["total_days_attempted"] += 1
                try:
                    raw = fut.result()
                except Exception as e:
                    logger.error(f"day {d} failed: {e}")
                    continue
                if raw is None:
                    summary["total_days_missing"] += 1
                    continue
                summary["total_days_present"] += 1
                by_ticker = parse_day(raw, target_set)
                for t, rows in by_ticker.items():
                    pair = by_massive[t]
                    if d < pair.effective_start:
                        continue
                    month_buffers[pair.hg_symbol].extend(rows)

        # Flush per-symbol
        for hg_sym, rows in month_buffers.items():
            out_path = (
                root / "fx_quotes_raw" / f"symbol={hg_sym}"
                / f"year={year}" / f"month={month}" / "data.parquet"
            )
            if skip_existing and out_path.exists():
                summary["months_skipped_existing"] += 1
                logger.info(f"[skip-existing] {hg_sym} {year}-{month}")
                continue
            n = rows_to_parquet(rows, out_path)
            summary["rows_per_symbol"][hg_sym] += n
            summary["months_written"] += 1
            logger.info(f"[wrote] {hg_sym} {year}-{month}: {n:,} rows")

    return summary
