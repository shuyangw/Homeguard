"""Massive (Polygon) flat-file FX downloader.

Reads daily CSV.gz aggregates from Massive's S3 bucket (one file = ALL pairs
for one trading day), filters to target tickers, writes per-symbol-per-month
parquet matching the existing `fx_1min/` canonical layout.

Auth via MASSIVE_S3_* env vars in .env (separate from MASSIVE_API_KEY which is
REST-only). See `docs/reference/DATA_INVENTORY.md` for layout reference.

Design rationale:
  - Source data is per-day-all-pairs, not per-symbol-per-day. Subclassing
    BaseDownloader (which iterates symbols) would re-download the same 15 MB
    file once per target ticker. Instead this module exposes a single-pass
    downloader that visits each day once and fans out to per-symbol buffers.
  - One month at a time held in memory; flushed atomically after the last day.
  - Boto3 ThreadPoolExecutor parallelizes per-day downloads within a month
    (typical ~22 trading days × 15 MB = ~330 MB peak per worker batch).

Schema mapping (Massive CSV column -> Homeguard canonical):
  ticker         -> drop (we filter by it)
  volume         -> volume
  open           -> open
  close          -> close
  high           -> high
  low            -> low
  window_start   -> timestamp (ns Unix -> Datetime[ns, UTC])
  transactions   -> trade_count
  (missing)      -> vwap   # set to close; documented as approximation
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

from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUCKET = "flatfiles"
PREFIX = "global_forex/minute_aggs_v1"


@dataclass(frozen=True)
class TargetPair:
    """One pair to acquire."""
    hg_symbol: str           # canonical Homeguard symbol e.g. "USDNOK"
    massive_ticker: str      # CSV ticker e.g. "C:USD-NOK"
    effective_start: date    # earliest date this pair has reliable coverage


def _env(name: str) -> str:
    """Read env var; raise if not set."""
    val = os.environ.get(name)
    if val:
        return val
    # Fallback: read from project .env if env var isn't already loaded
    proj = Path(__file__).resolve().parents[4]
    env_path = proj / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not in environment or .env")


def make_s3_client():
    """Construct an S3 client pointed at the Massive flat-file endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=_env("MASSIVE_S3_ENDPOINT"),
        aws_access_key_id=_env("MASSIVE_S3_ACCESS_KEY"),
        aws_secret_access_key=_env("MASSIVE_S3_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )


def key_for(d: date) -> str:
    """S3 key for a daily flat file."""
    return f"{PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"


def fetch_day(s3, day: date) -> bytes | None:
    """Download one daily CSV.gz. Returns the raw bytes or None if not present
    (weekends / holidays / pre-archive-start dates)."""
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


def parse_day(raw_gz: bytes, target_tickers: set[str]) -> dict[str, list[dict]]:
    """Decompress + parse one daily file. Returns {massive_ticker: [row, ...]}.

    Rows are emitted in the order they appear in the source file. The source
    is already grouped by ticker and sorted by window_start, so downstream
    sort-by-timestamp is cheap per group.
    """
    text = gzip.decompress(raw_gz).decode("utf-8")
    out: dict[str, list[dict]] = {}
    lines = text.splitlines()
    # Skip header line
    for line in lines[1:]:
        # ticker,volume,open,close,high,low,window_start,transactions
        comma1 = line.find(",")
        if comma1 < 0:
            continue
        ticker = line[:comma1]
        if ticker not in target_tickers:
            continue
        try:
            fields = line.split(",")
            ts_ns = int(fields[6])
            row = {
                "timestamp": ts_ns,
                "open": float(fields[2]),
                "high": float(fields[4]),
                "low": float(fields[5]),
                "close": float(fields[3]),
                "volume": int(fields[1]),
                "trade_count": int(fields[7]),
                # vwap absent in flat-file; set to close as documented approximation
                "vwap": float(fields[3]),
            }
        except (ValueError, IndexError) as e:
            logger.warning(f"skipping malformed line for {ticker}: {e!r}")
            continue
        out.setdefault(ticker, []).append(row)
    return out


def rows_to_parquet(rows: list[dict], out_path: Path) -> int:
    """Write rows to a parquet file at out_path, matching the existing canonical
    fx_1min schema (8 cols, Datetime[ns, UTC] + Float64×5 + Int64×2 + Float64).
    Atomic via tmp -> rename.
    """
    if not rows:
        return 0
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="ns", time_zone="UTC")),
    ).select(CANONICAL_OHLCV_SCHEMA).sort("timestamp")

    # Drop dups within month (shouldn't happen, but be safe)
    n_before = df.height
    df = df.unique(subset=["timestamp"], keep="last")
    if df.height != n_before:
        logger.warning(f"dropped {n_before - df.height} duplicate timestamps in {out_path}")

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
    concurrency: int = 8,
    skip_existing: bool = True,
) -> dict:
    """Bulk-download all `pairs` over [start_date, end_date], writing per-symbol
    per-month parquet under {storage_root}/fx_1min/symbol={SYM}/year={Y}/month={M}/data.parquet.

    Returns a dict summary: total_files_downloaded, missing_days, rows_per_symbol.

    skip_existing: if True, monthly parquet that already exists is left untouched
    (downloads still happen for those days but writes are skipped).
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
        # Per-pair buffers for this month
        month_buffers: dict[str, list[dict]] = defaultdict(list)

        # Filter days by per-pair effective_start downstream; download once for all
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_day = {
                ex.submit(fetch_day, s3, d): d for d in days
            }
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
                    # Honor effective_start
                    if d < pair.effective_start:
                        continue
                    month_buffers[pair.hg_symbol].extend(rows)

        # Flush month buffers per symbol
        for hg_sym, rows in month_buffers.items():
            out_path = (
                root / "fx_1min" / f"symbol={hg_sym}"
                / f"year={year}" / f"month={month}" / "data.parquet"
            )
            if skip_existing and out_path.exists():
                summary["months_skipped_existing"] += 1
                logger.info(f"[skip-existing] {hg_sym} {year}-{month}")
                continue
            n = rows_to_parquet(rows, out_path)
            summary["rows_per_symbol"][hg_sym] += n
            summary["months_written"] += 1
            logger.info(f"[wrote] {hg_sym} {year}-{month}: {n:,} rows -> {out_path}")

    return summary
