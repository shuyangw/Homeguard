"""Resample fx_1min/ to 17:00-ET-anchored daily bars, cached to fx_daily/.

FX trades 24/5; the market-convention day boundary is 17:00 America/New_York
(Sunday 17:00 -> Friday 17:00). Each minute is assigned to the FX trading day
whose (prev-day 17:00, this-day 17:00] window contains it; the daily close is
the last minute close inside that window. A +7h wall-clock shift after tz
conversion (24 - 17 = 7) maps each ET timestamp onto its FX date.

DST is handled by the America/New_York tz conversion; the +7h shift is applied
in local wall-clock time, so the boundary tracks 17:00 ET across DST changes.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger


def resample_fx_minute_to_daily(df_min: pd.DataFrame) -> pd.DataFrame:
    if df_min.empty:
        return pd.DataFrame(columns=["close"])
    ts_et = df_min["timestamp"].dt.tz_convert("America/New_York")
    fx_date = (ts_et + pd.Timedelta(hours=7)).dt.date
    tmp = df_min.assign(fx_date=fx_date).sort_values("timestamp")
    daily = tmp.groupby("fx_date").agg(close=("close", "last"))
    daily.index.name = "fx_date"
    return daily


def build_fx_daily_cache(pairs: list[str], start: date, end: date,
                         src_root: Path | None = None,
                         out_root: Path | None = None) -> list[str]:
    base = Path(get_local_storage_dir())
    src_root = src_root or (base / "fx_1min")
    out_root = out_root or (base / "fx_daily")
    written: list[str] = []
    for pair in pairs:
        sym_dir = src_root / f"symbol={pair}"
        if not sym_dir.exists():
            logger.warning(f"[build_fx_daily_cache] no source data for {pair}")
            continue
        lf = pl.scan_parquet(sym_dir / "**/*.parquet").select(["timestamp", "close"])
        pdf = lf.collect().to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
        pdf = pdf[(pdf["timestamp"].dt.date >= start) & (pdf["timestamp"].dt.date <= end)]
        daily = resample_fx_minute_to_daily(pdf)
        if daily.empty:
            continue
        out = daily.reset_index()
        out["year"] = pd.to_datetime(out["fx_date"]).dt.year
        out["month"] = pd.to_datetime(out["fx_date"]).dt.month
        for (yr, mo), grp in out.groupby(["year", "month"]):
            dst = out_root / f"symbol={pair}" / f"year={yr}" / f"month={mo}"
            dst.mkdir(parents=True, exist_ok=True)
            pl.from_pandas(grp[["fx_date", "close"]]).write_parquet(dst / "data.parquet")
        written.append(pair)
        logger.info(f"[build_fx_daily_cache] wrote {pair}: {len(daily)} daily bars")
    return written


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build fx_daily/ cache from fx_1min/")
    p.add_argument("--csv", required=True, help="Universe CSV with a 'symbol' column")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    pairs = pd.read_csv(args.csv)["symbol"].tolist()
    written = build_fx_daily_cache(
        pairs, date.fromisoformat(args.start), date.fromisoformat(args.end))
    logger.success(f"[build_fx_daily_cache] wrote {len(written)} pairs")


if __name__ == "__main__":
    main()
