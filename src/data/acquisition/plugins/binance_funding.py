"""Binance USDT-M perpetual funding-rate downloader.

Funding settles every 8h (3 events/UTC-day). We store the raw events plus a
daily-annualized funding series (sum the day's events x 365). Writes to
alt_data/funding/<root>/funding.parquet. Funding is realized (past) at use, so
it is causal for a next-day tilt."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_ROOT_TO_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def parse_funding(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"funding_time": pl.Datetime, "funding_rate": pl.Float64})
    return pl.DataFrame({
        "funding_time": [datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc).replace(tzinfo=None)
                         for r in rows],
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })


def daily_annualized(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(schema={"date": pl.Date, "funding_annualized": pl.Float64})
    return (df.with_columns(pl.col("funding_time").dt.date().alias("date"))
              .group_by("date").agg(pl.col("funding_rate").sum().alias("daily_funding"))
              .with_columns((pl.col("daily_funding") * 365.0).alias("funding_annualized"))
              .select("date", "funding_annualized").sort("date"))


class BinanceFundingPlugin:
    def __init__(self, storage_root: Path | None = None) -> None:
        self._root = storage_root or (get_local_storage_dir() / "alt_data")

    def fetch_symbol(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        out, cursor = [], start_ms
        while cursor < end_ms:
            r = requests.get(FUNDING_URL, params={"symbol": symbol, "startTime": cursor,
                                                  "endTime": end_ms, "limit": 1000}, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            last = batch[-1]["fundingTime"]
            if last <= cursor:
                break
            cursor = last + 1
            time.sleep(0.2)  # rate-limit courtesy
        return out

    def build(self, root_to_symbol: dict | None = None,
              start: str = "2019-09-01", end: str | None = None,
              *, skip_existing: bool = True) -> dict:
        root_to_symbol = root_to_symbol or _ROOT_TO_SYMBOL
        start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int((datetime.now(timezone.utc)).timestamp() * 1000) if end is None else \
                 int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000)
        summary = {}
        for root, symbol in root_to_symbol.items():
            out = self._root / "funding" / root / "funding.parquet"
            if skip_existing and out.exists():
                summary[root] = "skipped"
                continue
            try:
                rows = self.fetch_symbol(symbol, start_ms, end_ms)
            except requests.exceptions.RequestException as e:
                logger.warning(f"  {root} ({symbol}): fetch failed: {e}")
                summary[root] = f"error: {e}"
                continue
            df = daily_annualized(parse_funding(rows))
            if df.height:
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(out.suffix + ".tmp")
                df.write_parquet(tmp)
                os.replace(tmp, out)
                (out.parent / "_snapshot.json").write_text(json.dumps(
                    {"fetched_utc": datetime.now(timezone.utc).date().isoformat(), "rows": df.height}))
                summary[root] = f"wrote {df.height} daily rows"
            else:
                summary[root] = "no data"
        return summary
