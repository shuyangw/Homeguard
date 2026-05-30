"""Validate a SIP-pass dataset (raw or split) against OHLCV invariants.

Checks per symbol:
  - OHLCV invariants: low <= min(open,close); high >= max(open,close); volume >= 0; no NaN
  - Timestamp monotonicity (per partition)
  - Bar count sanity (flag days with < 100 bars)
  - Zero-row detection

Outputs:
  - output/data_validation/<subdir>_coverage.csv
  - output/data_validation/<subdir>_low_bar_days.csv
  - Updates _manifests/<subdir>.status.csv with validation_error column

Usage:
    python scripts/data/validate_sip_dataset.py --pass raw
    python scripts/data/validate_sip_dataset.py --pass split --revalidate
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.acquisition.status_tracker import (
    rebuild_tracker,
    write_tracker_csv,
)
from src.settings import (
    EQUITIES_SIP_RAW_1MIN,
    EQUITIES_SIP_SPLIT_1MIN,
    get_local_storage_dir,
    get_output_dir,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PASS_TO_SUBDIR = {
    "raw": EQUITIES_SIP_RAW_1MIN,
    "split": EQUITIES_SIP_SPLIT_1MIN,
}

LOW_BAR_THRESHOLD = 100


def validate_symbol(base_dir: Path, subdir: str, symbol: str) -> tuple[
    str | None, list[dict], dict,
]:
    """Validate one symbol's partitions.

    Returns: (validation_error or None, low_bar_day_rows, coverage_row)
    """
    sym_dir = base_dir / subdir / f"symbol={symbol}"
    coverage = {
        "symbol": symbol, "first_date": None, "last_date": None,
        "trading_days": 0, "total_bars": 0,
    }
    if not sym_dir.exists():
        return ("missing_directory", [], coverage)

    partition_paths = sorted(sym_dir.glob("year=*/month=*/data.parquet"))
    if not partition_paths:
        return ("no_partitions", [], coverage)

    low_bar_rows: list[dict] = []
    error: str | None = None
    all_dates: set = set()
    first_ts = None
    last_ts = None
    total_bars = 0

    for p in partition_paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        # NaN check
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if df[required].isna().any().any():
            error = "nan_in_required_columns"
            break
        # Timestamp monotonicity
        ts = pd.to_datetime(df["timestamp"])
        if not ts.is_monotonic_increasing:
            error = "non_monotonic_timestamps"
            break
        # OHLCV invariants
        if not (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all():
            error = "low_greater_than_open_or_close"
            break
        if not (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all():
            error = "high_less_than_open_or_close"
            break
        if (df["volume"] < 0).any():
            error = "negative_volume"
            break
        # Low-bar-day flagging
        per_day = ts.dt.date.value_counts()
        low_days = per_day[per_day < LOW_BAR_THRESHOLD]
        for day, count in low_days.items():
            low_bar_rows.append({
                "symbol": symbol, "date": str(day), "bars": int(count),
                "partition": str(p),
            })

        all_dates.update(ts.dt.date.unique())
        local_first, local_last = ts.min(), ts.max()
        first_ts = local_first if first_ts is None or local_first < first_ts else first_ts
        last_ts = local_last if last_ts is None or local_last > last_ts else last_ts
        total_bars += len(df)

    coverage.update({
        "first_date": str(first_ts.date()) if first_ts is not None else None,
        "last_date": str(last_ts.date()) if last_ts is not None else None,
        "trading_days": len(all_dates),
        "total_bars": total_bars,
    })
    return (error, low_bar_rows, coverage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pass", dest="feed_pass", choices=("raw", "split"), required=True,
    )
    parser.add_argument(
        "--revalidate", action="store_true",
        help="Re-validate symbols even if already in coverage CSV",
    )
    args = parser.parse_args(argv)
    subdir = PASS_TO_SUBDIR[args.feed_pass]
    base_dir = get_local_storage_dir()

    pass_dir = base_dir / subdir
    if not pass_dir.exists():
        logger.error(f"Pass directory does not exist: {pass_dir}")
        return 1

    symbols = sorted(
        p.name.replace("symbol=", "")
        for p in pass_dir.glob("symbol=*")
    )
    logger.info(f"[{subdir}] Validating {len(symbols)} symbols")

    val_dir = get_output_dir() / "data_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = val_dir / f"{subdir}_coverage.csv"
    low_path = val_dir / f"{subdir}_low_bar_days.csv"

    already_done: set = set()
    if coverage_path.exists() and not args.revalidate:
        already_done = set(
            pd.read_csv(coverage_path)["symbol"].astype(str)
        )
        logger.info(f"Skipping {len(already_done)} already-validated symbols")

    validation_errors: dict[str, str] = {}
    coverage_rows: list[dict] = []
    low_bar_rows: list[dict] = []

    for i, symbol in enumerate(symbols, 1):
        if symbol in already_done:
            continue
        err, lows, cov = validate_symbol(base_dir, subdir, symbol)
        if err:
            validation_errors[symbol] = err
        coverage_rows.append(cov)
        low_bar_rows.extend(lows)
        if i % 500 == 0:
            logger.info(f"  ...validated {i}/{len(symbols)}")

    # Append-mode CSV
    if coverage_rows:
        cov_df = pd.DataFrame(coverage_rows)
        cov_df.to_csv(
            coverage_path, mode="a", header=not coverage_path.exists(),
            index=False,
        )
        logger.info(f"Coverage written: {coverage_path}")
    if low_bar_rows:
        low_df = pd.DataFrame(low_bar_rows)
        low_df.to_csv(
            low_path, mode="a", header=not low_path.exists(),
            index=False,
        )
        logger.info(f"Low-bar days written: {low_path}")

    # Update tracker with validation errors
    tracker_rows = rebuild_tracker(
        base_dir=base_dir, subdir=subdir, universe=symbols,
        validation_errors=validation_errors,
    )
    tracker_path = base_dir / "_manifests" / f"{subdir}.status.csv"
    write_tracker_csv(tracker_path, tracker_rows)

    bad = sum(1 for e in validation_errors.values())
    logger.info(
        f"[{subdir}] Validation done. {bad} symbols flagged as broken. "
        f"Tracker: {tracker_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
