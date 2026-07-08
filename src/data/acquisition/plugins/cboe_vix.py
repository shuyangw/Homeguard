"""Cboe/CFE VX (VIX futures) daily settlement -> front/second continuous curve.

Fetches per-contract VX daily settlement from Cboe's public historical data,
then derives VX1 (nearest unexpired) and VX2 (next) per date. The exact Cboe
endpoint/column names are confirmed against the live source in the build step;
the roll logic (build_front_second) is deterministic and unit-tested offline.
Writes alt_data/vix/vx_curve.parquet."""
from __future__ import annotations

import io
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Confirmed live 2026-07-07: Cboe publishes per-contract VX daily settlement
# history keyed by the contract's expiration (settlement) date, e.g.
# .../VX/VX_2025-01-22.csv for the January 2025 contract (expiry 2025-01-22).
# A missing/never-listed contract returns HTTP 403 (S3-style AccessDenied),
# not 404 -- callers must treat any non-200 as "skip this contract".
CBOE_VX_BASE = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/"

# Live per-contract CSV columns (confirmed): "Trade Date,Futures,Open,High,Low,
# Close,Settle,Change,Total Volume,EFP,Open Interest". There is no expiration
# column -- the expiry is implicit in which file was requested, so callers
# pass it in explicitly via the `expiry` argument.
_VX_COLMAP = {"Trade Date": "date_raw", "Settle": "settle"}

_EARLIEST_CONTRACT_YEAR = 2013


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    fridays_seen = 0
    while True:
        if d.weekday() == 4:
            fridays_seen += 1
            if fridays_seen == 3:
                return d
        d += timedelta(days=1)


def vx_monthly_expiry(year: int, month: int) -> date:
    """VX monthly expiry: the Wednesday 30 days before the third Friday of the
    following calendar month. Does NOT apply Cboe holiday shifts (rare edge
    case); a contract missed by +/-1 day here simply fails to fetch and is
    skipped, it is not fabricated."""
    next_month, next_year = (1, year + 1) if month == 12 else (month + 1, year)
    return _third_friday(next_year, next_month) - timedelta(days=30)


def _candidate_expiries(today: date | None = None) -> list[date]:
    today = today or datetime.now(timezone.utc).date()
    end_year = today.year + 2
    return [vx_monthly_expiry(y, m) for y in range(_EARLIEST_CONTRACT_YEAR, end_year + 1) for m in range(1, 13)]


def parse_vx_settlement(raw: bytes, expiry: date | None = None) -> pl.DataFrame:
    """Parse a Cboe VX per-contract settlement CSV into (date, expiry, settle).

    `expiry` is the contract's expiration date -- the live Cboe CSV has no
    expiration column, so the caller (who requested this specific contract's
    file) supplies it. Settle values of 0 are a known data-quality artifact of
    the earliest (2013) contracts and are dropped, not treated as real quotes.
    """
    empty = pl.DataFrame(schema={"date": pl.Date, "expiry": pl.Date, "settle": pl.Float64})
    if not raw:
        return empty
    df = pl.read_csv(io.BytesIO(raw), ignore_errors=True)
    df = df.rename({k: v for k, v in _VX_COLMAP.items() if k in df.columns})
    if not {"date_raw", "settle"}.issubset(set(df.columns)):
        logger.warning(f"[cboe_vix] settlement CSV missing expected columns: {df.columns}")
        return empty
    df = df.with_columns(
        pl.col("date_raw").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("date"),
        pl.col("settle").cast(pl.Float64, strict=False),
    )
    if expiry is not None:
        df = df.with_columns(pl.lit(expiry).alias("expiry"))
    elif "expiry_raw" in df.columns:
        df = df.with_columns(
            pl.col("expiry_raw").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("expiry"))
    else:
        logger.warning("[cboe_vix] no expiry provided and no expiry column in CSV")
        return empty
    return df.filter(pl.col("settle") > 0).select("date", "expiry", "settle").drop_nulls()


def build_front_second(per_contract: pl.DataFrame) -> pl.DataFrame:
    """Per date, pick the nearest UNEXPIRED contract (VX1) and the next (VX2)."""
    rows = []
    for d, grp in per_contract.filter(pl.col("expiry") > pl.col("date")).group_by("date"):
        g = grp.sort("expiry")
        if g.height < 2:
            continue
        date_val = d[0] if isinstance(d, tuple) else d
        v1, v2 = g.row(0, named=True), g.row(1, named=True)
        rows.append({"date": date_val, "vx1_settle": v1["settle"], "vx2_settle": v2["settle"],
                     "vx1_dte": (v1["expiry"] - date_val).days})
    return pl.DataFrame(rows).sort("date") if rows else pl.DataFrame(
        schema={"date": pl.Date, "vx1_settle": pl.Float64, "vx2_settle": pl.Float64, "vx1_dte": pl.Int64})


class CboeVixPlugin:
    def __init__(self, storage_root: Path | None = None) -> None:
        self._root = storage_root or (get_local_storage_dir() / "alt_data")

    def fetch_all_contracts(self) -> pl.DataFrame:
        """Fetch every VX contract's settlement history and concat.

        Enumerates candidate monthly expiries from 2013 to ~2 years out,
        fetches each contract's CSV, and concats. Cboe shifts the expiry to
        the preceding business day around holidays, so the formulaic date is
        also tried +/-1 day if the primary guess misses. A never-listed/
        not-yet-open contract returns HTTP 403/404 on all three attempts --
        that contract is logged and skipped, never fabricated. Session
        timeout keeps a single slow/hung request from stalling the build."""
        session = requests.Session()
        frames: list[pl.DataFrame] = []
        skipped = 0
        for expiry in _candidate_expiries():
            parsed = None
            for candidate in (expiry, expiry - timedelta(days=1), expiry + timedelta(days=1)):
                url = f"{CBOE_VX_BASE}VX_{candidate.isoformat()}.csv"
                try:
                    resp = session.get(url, timeout=15)
                except requests.RequestException as exc:
                    logger.warning(f"[cboe_vix] request failed for {candidate}: {exc}")
                    continue
                if resp.status_code != 200:
                    continue
                p = parse_vx_settlement(resp.content, expiry=candidate)
                if p.height > 0:
                    parsed = p
                    break
            if parsed is not None:
                frames.append(parsed)
            else:
                skipped += 1
        logger.info(f"[cboe_vix] fetched {len(frames)} contracts, skipped {skipped} candidates")
        if not frames:
            raise ValueError("cboe_vix: no VX contracts fetched -- check CBOE_VX_BASE / network")
        return pl.concat(frames)

    def build(self, per_contract: pl.DataFrame | None = None) -> Path:
        pc = per_contract if per_contract is not None else self.fetch_all_contracts()
        curve = build_front_second(pc)
        out = self._root / "vix" / "vx_curve.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        curve.write_parquet(tmp)
        os.replace(tmp, out)
        (out.parent / "_snapshot.json").write_text(json.dumps(
            {"fetched_utc": datetime.now(timezone.utc).date().isoformat(), "rows": curve.height}))
        logger.info(f"[cboe_vix] wrote {curve.height} VX curve rows to {out}")
        return out
