"""Canonicalization layer for the equity-options minute store.

This module is the ONLY sanctioned entry point for reading the 1-minute options
store. It normalizes the on-disk vendor schema into a single canonical schema
and encodes three registered rulings that downstream code must not re-litigate.

--------------------------------------------------------------------------
LOUD WARNING 1 -- open/high/low/close/volume/trade_count/vwap are TRADE-DERIVED
--------------------------------------------------------------------------
Those columns describe PRINTS, not the market. They are NEVER usable as a mark.
An option can print once a day, at any point on (or outside) the spread, or not
at all. The ONLY sanctioned mark is `mid` = (bid + ask) / 2, and only where
`quote_valid` is True. If you find yourself marking a position off `close`,
stop -- that is a bug, not a shortcut.

--------------------------------------------------------------------------
LOUD WARNING 2 -- `gamma_eod` / `oi_eod` are LEAK-BEARING at the snapshot minute
--------------------------------------------------------------------------
`scripts/data/combine_options_data.py` derives a `_date` by slicing the intraday
timestamp (~line 179) and left-joins the END-OF-DAY gamma/open-interest frame on
that same date (~line 239). Measured on data: OI is constant across all minutes
of a session (0 of 18,879 contract-sessions vary) and changes day to day.

Therefore session t's rows carry session t's END-OF-DAY open interest, which is
not published until the following morning. Reading `oi_eod` / `gamma_eod` at the
15:45 snapshot on session t is a HARD LOOKAHEAD LEAK.

REGISTERED RULING (V6): every use of `oi_eod` / `gamma_eod` must lag by at least
ONE TRADING SESSION. Use `add_eod_lag()` and consume `oi_eod_lag1` /
`gamma_eod_lag1`. The raw same-session columns are retained -- deliberately with
their `_eod` suffix intact -- so the constraint stays visible at every call site.

NOTE: `src/strategies/options/data_loader.py:20-26` renames
`gamma_eod -> gamma` and `open_interest_eod -> open_interest`, which STRIPS the
marker. This module never does that. `open_interest_eod -> oi_eod` is allowed
(suffix preserved); dropping `_eod` is not.

--------------------------------------------------------------------------
LOUD WARNING 3 -- the snapshot minute is 15:45 ET and lives HERE
--------------------------------------------------------------------------
`SNAPSHOT_TIME_ET` is the single source of truth for option marks, signal
truncation, and daily hedge execution. Nothing downstream may slice bars itself.
`src/strategies/options/data_loader.py:51` `get_eod_chain()` hardcodes 16:00 --
that is WRONG for this work; do not reuse it.

--------------------------------------------------------------------------
Timestamps
--------------------------------------------------------------------------
On disk `timestamp` is a TZ-NAIVE ISO-8601 STRING. Verified across both 2024 DST
transitions (SPY 2024-03 and 2024-11): every session runs 09:30:00 -> 16:00:00
with 391 distinct minutes, unshifted across the boundary. The naive stamps are
therefore US/Eastern WALL CLOCK. They are localized to America/New_York and
converted to `Datetime[us, UTC]` (the repo standard).

--------------------------------------------------------------------------
Validity
--------------------------------------------------------------------------
`quote_valid` (V3, registered): bid > 0 AND ask >= bid AND both finite.
Rows failing the rule are FLAGGED, never dropped silently and never repaired.
There is NO imputation, forward-fill, interpolation or smoothing anywhere in
this module. `mid` / `spread_abs` / `spread_rel` are NULL exactly where
`quote_valid` is False.
"""

from __future__ import annotations

import argparse
import os
import time as _pytime
from datetime import date, time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import polars as pl

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Registered constants
# --------------------------------------------------------------------------

EASTERN_TZ = "America/New_York"

#: The registered snapshot minute. Single source of truth -- do not parameterize
#: this away, and do not slice bars anywhere else.
SNAPSHOT_TIME_ET: time = time(15, 45)

SOURCE_STORE_NAME = "options_combined"
EOD_STORE_NAME = "options_chain_eod"

#: Divergence D2: on disk `right` is "PUT"/"CALL", not "P"/"C" as the spec assumed.
RIGHT_MAP = {"PUT": "P", "CALL": "C"}

CONTRACT_KEY = ["root", "expiry", "strike", "right"]

CANONICAL_COLUMNS: List[str] = [
    "root",
    "ts",
    "session_date",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "mid",
    "spread_abs",
    "spread_rel",
    "quote_valid",
    "dte",
    "dte_trading",
    # trade-derived -- NEVER a mark (see module docstring)
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    # vendor-computed, caveated (not recomputed here)
    "implied_vol",
    "delta",
    "theta",
    "vega",
    "underlying_px",
    # leak-bearing at the snapshot minute -- `_eod` suffix is load-bearing
    "gamma_eod",
    "oi_eod",
]

EOD_EXTRA_COLUMNS: List[str] = ["snapshot_ts", "snapshot_fallback"]

_SOURCE_COLUMNS = [
    "timestamp",
    "expiration",
    "strike",
    "right",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "bid_close",
    "ask_close",
    "implied_vol",
    "delta",
    "theta",
    "vega",
    "underlying_px",
    "gamma_eod",
    "open_interest_eod",
]


class UnexpectedRightError(ValueError):
    """Raised when `right` holds a value outside {PUT, CALL}. Fail loud."""


# --------------------------------------------------------------------------
# Trading calendar (reuses the repo's NYSE calendar -- no forked calendar)
# --------------------------------------------------------------------------

_TRADING_DAYS_CACHE: dict = {}


def _trading_day_ordinals(lo: date, hi: date) -> np.ndarray:
    """Sorted array of NYSE trading days (as `date.toordinal()`) covering [lo, hi]."""
    key = (lo.year - 1, hi.year + 1)
    cached = _TRADING_DAYS_CACHE.get(key)
    if cached is None:
        from src.backtesting.utils.market_calendar import MarketCalendar

        days = MarketCalendar().get_trading_days(
            f"{key[0]}-01-01", f"{key[1]}-12-31"
        )
        cached = np.array([d.date().toordinal() for d in days], dtype=np.int64)
        _TRADING_DAYS_CACHE[key] = cached
    return cached


def _sessions_through(days: Sequence[date], grid: np.ndarray) -> np.ndarray:
    """Count of trading sessions <= each day."""
    ords = np.array(
        [d.toordinal() if d is not None else -1 for d in days], dtype=np.int64
    )
    return np.searchsorted(grid, ords, side="right")


def trading_days_between(starts: Sequence[date], ends: Sequence[date]) -> np.ndarray:
    """Trading sessions strictly after `start` and up to and including `end`.

    Same-day expiry (0DTE) -> 0. Next trading session -> 1. Fri -> Mon -> 1.
    """
    if not len(starts):
        return np.array([], dtype=np.int64)
    known = [d for d in list(starts) + list(ends) if d is not None]
    grid = _trading_day_ordinals(min(known), max(known))
    return _sessions_through(ends, grid) - _sessions_through(starts, grid)


def _session_index(days: Sequence[date]) -> np.ndarray:
    """Position of each day on the NYSE trading-day grid (for adjacency checks)."""
    known = [d for d in days if d is not None]
    grid = _trading_day_ordinals(min(known), max(known))
    return _sessions_through(days, grid)


# --------------------------------------------------------------------------
# Table A -- options_chain_1m (canonicalized READ layer)
# --------------------------------------------------------------------------


def _finite(col: str) -> pl.Expr:
    return (
        pl.col(col).is_not_null()
        & pl.col(col).is_not_nan()
        & pl.col(col).is_finite()
    )


def _expiry_expr(df: pl.DataFrame) -> pl.Expr:
    """`expiration` ships as string in most partitions and as date32 in 100 of
    them (SPY 2017-2018, QQQ 2017/2018/2023). Accept both; fail loud otherwise."""
    dtype = df.schema["expiration"]
    if dtype == pl.Date:
        return pl.col("expiration").alias("expiry")
    if dtype in (pl.String, pl.Utf8):
        return pl.col("expiration").str.to_date().alias("expiry")
    raise TypeError(
        f"[-] unsupported `expiration` dtype {dtype}; expected String or Date"
    )


def canonicalize_frame(df: pl.DataFrame, root: str) -> pl.DataFrame:
    """Map one raw on-disk frame to the canonical schema.

    Pure and allocation-bounded: it never drops rows, never imputes, and never
    recomputes greeks or implied vol.
    """
    unexpected = set(df["right"].unique().to_list()) - set(RIGHT_MAP)
    if unexpected:
        raise UnexpectedRightError(
            f"[-] unexpected `right` value(s) {sorted(unexpected)} in root={root}; "
            f"expected one of {sorted(RIGHT_MAP)}"
        )

    quote_valid = (
        _finite("bid_close")
        & _finite("ask_close")
        & (pl.col("bid_close") > 0.0)
        & (pl.col("ask_close") >= pl.col("bid_close"))
    )

    out = df.with_columns(
        pl.lit(root, dtype=pl.String).alias("root"),
        pl.col("timestamp")
        .str.to_datetime(time_unit="us")
        .dt.replace_time_zone(EASTERN_TZ)
        .dt.convert_time_zone("UTC")
        .alias("ts"),
        pl.col("timestamp").str.slice(0, 10).str.to_date().alias("session_date"),
        _expiry_expr(df),
        pl.col("right").replace_strict(RIGHT_MAP, return_dtype=pl.String).alias("right"),
        pl.col("bid_close").alias("bid"),
        pl.col("ask_close").alias("ask"),
        pl.col("gamma_eod").alias("gamma_eod"),
        pl.col("open_interest_eod").alias("oi_eod"),
        quote_valid.alias("quote_valid"),
    )

    mid = pl.when(pl.col("quote_valid")).then(
        (pl.col("bid") + pl.col("ask")) / 2.0
    ).otherwise(None)
    spread_abs = pl.when(pl.col("quote_valid")).then(
        pl.col("ask") - pl.col("bid")
    ).otherwise(None)

    out = out.with_columns(mid.alias("mid"), spread_abs.alias("spread_abs"))
    out = out.with_columns(
        pl.when(pl.col("quote_valid") & (pl.col("mid") != 0.0))
        .then(pl.col("spread_abs") / pl.col("mid"))
        .otherwise(None)
        .alias("spread_rel"),
        (pl.col("expiry") - pl.col("session_date")).dt.total_days().cast(pl.Int32).alias("dte"),
    )

    out = out.with_columns(
        pl.Series(
            "dte_trading",
            trading_days_between(
                out["session_date"].to_list(), out["expiry"].to_list()
            ),
            dtype=pl.Int32,
        )
    )
    return out.select(CANONICAL_COLUMNS)


def _month_path(root: str, year: int, month: int, store: str = SOURCE_STORE_NAME) -> Path:
    return (
        get_local_storage_dir()
        / "options"
        / store
        / f"root={root}"
        / f"year={year:04d}"
        / f"month={month:02d}"
        / "data.parquet"
    )


def iter_canonical_batches(
    root: str,
    year: int,
    month: int,
    row_groups_per_batch: int = 1,
    max_batches: Optional[int] = None,
) -> Iterator[pl.DataFrame]:
    """Stream a root-month as canonical frames, one (group of) row group(s) at a time.

    This is a READ layer over the native store -- it never copies the 233 GB.
    """
    import pyarrow.parquet as pq

    path = _month_path(root, year, month)
    if not path.exists():
        raise FileNotFoundError(f"[-] no options data at {path}")

    pf = pq.ParquetFile(path)
    emitted = 0
    for start in range(0, pf.metadata.num_row_groups, row_groups_per_batch):
        idx = list(
            range(
                start,
                min(start + row_groups_per_batch, pf.metadata.num_row_groups),
            )
        )
        tbl = pf.read_row_groups(idx, columns=_SOURCE_COLUMNS)
        yield canonicalize_frame(pl.from_arrow(tbl), root=root)
        emitted += 1
        if max_batches is not None and emitted >= max_batches:
            return


# --------------------------------------------------------------------------
# The snapshot guard -- single source of truth
# --------------------------------------------------------------------------


def _snapshot_cutoff_expr(session_close_et: Optional[time]) -> pl.Expr:
    """UTC timestamp of the effective snapshot cutoff for each row's session.

    Divergence D3: on early-close sessions the vendor PADS bars out to 16:00 with
    zero volume and stale quotes. Passing `session_close_et` clamps the cutoff to
    the real close so a padded post-close bar can never become the mark.
    """
    cutoff = SNAPSHOT_TIME_ET
    if session_close_et is not None and session_close_et < SNAPSHOT_TIME_ET:
        cutoff = session_close_et
    return (
        pl.col("session_date")
        .cast(pl.Datetime("us"))
        .dt.offset_by(f"{cutoff.hour}h")
        .dt.offset_by(f"{cutoff.minute}m")
        .dt.replace_time_zone(EASTERN_TZ)
        .dt.convert_time_zone("UTC")
    )


def snapshot_from_bars(
    bars: pl.DataFrame, session_close_et: Optional[time] = None
) -> pl.DataFrame:
    """Return the snapshot record per (contract, session) from canonical bars.

    The bar at exactly 15:45:00 ET if it exists AND has a valid quote; otherwise
    the LATEST bar <= 15:45:00 ET with a valid quote, flagged
    `snapshot_fallback=True`. `snapshot_ts` always carries the ACTUAL bar time --
    never assumed, never backfilled to 15:45. If no valid bar <= the cutoff
    exists for a contract-session, NO record is emitted for it.

    LEAKAGE GUARD: a bar after the cutoff can never be returned.
    """
    if bars.height == 0:
        return bars.with_columns(
            pl.col("ts").alias("snapshot_ts"),
            pl.lit(False).alias("snapshot_fallback"),
        ).select(CANONICAL_COLUMNS + EOD_EXTRA_COLUMNS)

    keyed = CONTRACT_KEY + ["session_date"]

    picked = (
        bars.with_columns(
            _snapshot_cutoff_expr(session_close_et).alias("_cutoff"),
            _snapshot_cutoff_expr(None).alias("_registered"),
        )
        .filter(pl.col("quote_valid") & (pl.col("ts") <= pl.col("_cutoff")))
        .sort("ts")
        .group_by(keyed, maintain_order=False)
        .last()
    )

    # `snapshot_fallback` flags ANY row not taken at the registered 15:45 minute,
    # including rows clamped by an early close.
    return (
        picked.with_columns(
            pl.col("ts").alias("snapshot_ts"),
            (pl.col("ts") != pl.col("_registered")).alias("snapshot_fallback"),
        )
        .select(CANONICAL_COLUMNS + EOD_EXTRA_COLUMNS)
        .sort(keyed)
    )


# --------------------------------------------------------------------------
# The `_eod` lag primitive (V6)
# --------------------------------------------------------------------------


def add_eod_lag(df: pl.DataFrame) -> pl.DataFrame:
    """Add `oi_eod_lag1` / `gamma_eod_lag1` -- the PRIOR TRADING SESSION's values.

    Registered behaviour:
      * the lagged value is the same contract's value from the IMMEDIATELY
        PRECEDING NYSE trading session;
      * NULL on the contract's first observed session;
      * NULL when the immediately preceding trading session is absent for that
        contract -- a stale value is NEVER carried across a gap (no forward-fill,
        no imputation).

    The raw same-session `oi_eod` / `gamma_eod` are retained and remain
    leak-bearing at the snapshot minute -- see the module docstring (V6).
    """
    if df.height == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("oi_eod_lag1"),
            pl.lit(None, dtype=pl.Float64).alias("gamma_eod_lag1"),
        )

    out = df.sort(CONTRACT_KEY + ["session_date"])
    out = out.with_columns(
        pl.Series("_sidx", _session_index(out["session_date"].to_list()), dtype=pl.Int64)
    )
    adjacent = (
        pl.col("_sidx") - pl.col("_sidx").shift(1).over(CONTRACT_KEY)
    ) == 1
    return out.with_columns(
        pl.when(adjacent)
        .then(pl.col("oi_eod").shift(1).over(CONTRACT_KEY))
        .otherwise(None)
        .alias("oi_eod_lag1"),
        pl.when(adjacent)
        .then(pl.col("gamma_eod").shift(1).over(CONTRACT_KEY))
        .otherwise(None)
        .alias("gamma_eod_lag1"),
    ).drop("_sidx")


# --------------------------------------------------------------------------
# Table B -- options_chain_eod (derived, MATERIALIZED)
# --------------------------------------------------------------------------


def _early_closes(year: int, month: int) -> dict:
    """Map of session_date -> early close time (ET) for sessions closing < 15:45."""
    from src.backtesting.utils.market_calendar import MarketCalendar

    cal = MarketCalendar().calendar
    start = date(year, month, 1)
    end = date(year + (month == 12), (month % 12) + 1, 1)
    sched = cal.schedule(start_date=str(start), end_date=str(end))
    out = {}
    for ts, close in zip(sched.index, sched["market_close"]):
        local = close.tz_convert(EASTERN_TZ).time()
        if local < SNAPSHOT_TIME_ET:
            out[ts.date()] = local
    return out


def build_chain_eod_frame(root: str, year: int, month: int) -> pl.DataFrame:
    """Build the in-memory options_chain_eod frame for one root-month.

    Streams the source month row-group by row-group, keeping only bars at or
    before the effective snapshot cutoff, then applies the section-1 guard.

    CAVEAT: `oi_eod_lag1` / `gamma_eod_lag1` are computed WITHIN this month, so
    the month's first session always carries NULL lags. Consumers spanning a
    month boundary must re-run `add_eod_lag()` over the concatenated frame --
    this module will never forward-fill across the seam to hide it.
    """
    early = _early_closes(year, month)
    parts = []
    n_rows = 0
    n_invalid = 0
    for batch in iter_canonical_batches(root, year, month, row_groups_per_batch=8):
        n_rows += batch.height
        n_invalid += int((~batch["quote_valid"]).sum())
        if not early:
            kept = snapshot_from_bars(batch)
        else:
            kept = pl.concat(
                [
                    snapshot_from_bars(sub, session_close_et=early.get(d))
                    for (d,), sub in batch.partition_by(
                        "session_date", as_dict=True
                    ).items()
                ],
                how="vertical",
            )
        if kept.height:
            parts.append(kept)

    if not parts:
        logger.warning(f"[!] no snapshot rows for root={root} {year}-{month:02d}")
        return pl.DataFrame(schema={c: pl.Null for c in CANONICAL_COLUMNS})

    # Row groups can straddle a contract-session, so re-apply the guard globally.
    merged = pl.concat(parts, how="vertical")
    keyed = CONTRACT_KEY + ["session_date"]
    merged = (
        merged.sort("snapshot_ts")
        .group_by(keyed, maintain_order=False)
        .last()
        .sort(keyed)
    )
    merged = add_eod_lag(merged)
    logger.info(
        f"[+] {root} {year}-{month:02d}: scanned {n_rows:,} bars "
        f"({n_invalid:,} quote_valid=False, flagged not dropped) -> "
        f"{merged.height:,} snapshot rows, "
        f"{int(merged['snapshot_fallback'].sum()):,} fallbacks"
    )
    return merged


def build_chain_eod(
    root: str, year: int, month: int, overwrite: bool = False
) -> Optional[Path]:
    """Materialize options_chain_eod for one root-month as hive-partitioned parquet.

    Layout mirrors the source store:
        <storage>/options/options_chain_eod/root=X/year=YYYY/month=MM/data.parquet
    """
    out_path = _month_path(root, year, month, store=EOD_STORE_NAME)
    if out_path.exists() and not overwrite:
        logger.info(f"[+] exists, skipping -> {out_path}")
        return out_path

    df = build_chain_eod_frame(root, year, month)
    if df.height == 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        f"[+] wrote {df.height:,} rows -> {out_path} "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )
    return out_path


def _available_root_months(root: str) -> List[tuple]:
    base = get_local_storage_dir() / "options" / SOURCE_STORE_NAME / f"root={root}"
    found = []
    for ydir in sorted(base.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            if (mdir / "data.parquet").exists():
                found.append(
                    (int(ydir.name.split("=")[1]), int(mdir.name.split("=")[1]))
                )
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from src.utils.run_status import RunStatus

    ap = argparse.ArgumentParser(description="Build the options_chain_eod table.")
    ap.add_argument("--root", required=True, help="Option root, e.g. SPY")
    ap.add_argument("--year", type=int, help="Year (omit for all available)")
    ap.add_argument("--month", type=int, help="Month (omit for all in year)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    months = _available_root_months(args.root)
    if args.year is not None:
        months = [m for m in months if m[0] == args.year]
    if args.month is not None:
        months = [m for m in months if m[1] == args.month]
    if not months:
        logger.error(f"[-] no source months matched root={args.root}")
        return 1

    with RunStatus(
        "options_chain_eod",
        meta={"root": args.root, "months": len(months)},
    ) as status:
        for i, (y, m) in enumerate(months, start=1):
            build_chain_eod(args.root, y, m, overwrite=args.overwrite)
            status.heartbeat(note=f"{args.root} {y}-{m:02d} ({i}/{len(months)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
