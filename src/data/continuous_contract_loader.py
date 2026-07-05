"""Continuous futures contract loader with 3 adjustment methods.

Reads raw .v.0 continuous bars from futures/databento/1min/ and per-contract
bars from futures/databento/per_contract_1min/. Provides:
- load(symbol, method) -> pl.DataFrame for raw | ratio_adjusted | panama_adjusted
- detect_roll_dates(symbol) -> list[date]
- aggregate_to_daily(symbol, method) -> pl.DataFrame
- aggregate_to_hourly(symbol, method) -> pl.DataFrame

Roll detection: per-day highest-volume outright contract. Spreads
(symbols containing "-") are excluded from active-contract candidates.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.data.futures.paths import continuous_1min_dir, per_contract_1min_dir

# CME month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
_MONTH_CODES = "FGHJKMNQUVXZ"

# Process-wide cache of the per-(root, calendar year) DAILY (date, symbol,
# vol) table -- NOT the raw per-contract minute bars. `_active_contract_per_day`
# is called once per (root, window) by every `load_daily_panel`/walk-forward
# window; a multi-window walk-forward's windows overlap heavily (e.g. Carver
# walk-forward's rolling train segment), so without caching the SAME
# per-contract minute-bar years get re-read and re-scanned from disk
# repeatedly. Caching is done at YEAR granularity (not "full history per
# root") deliberately: caching full multi-decade history per root would hold
# the raw multi-GB minute-bar data for every root in memory simultaneously
# (observed: >40GB resident for a 12-root universe) -- caching small
# per-year DAILY aggregates instead bounds memory to O(roots x years) tiny
# tables, while the large raw per-file minute data is transient (freed right
# after each file's groupby-aggregate).
_YEAR_DAILY_VOLUME_CACHE: dict[tuple[str, str, int], pl.DataFrame] = {}


def _is_outright(symbol: str, root: str) -> bool:
    """True if `symbol` is an outright contract of `root` (not a spread)."""
    if "-" in symbol or " " in symbol:
        return False
    if not symbol.startswith(root):
        return False
    suffix = symbol[len(root):]
    if len(suffix) < 2:
        return False
    # suffix should be one month-code letter + digits (year)
    return suffix[0] in _MONTH_CODES and suffix[1:].isdigit()


def _outright_filter_expr(root: str) -> pl.Expr:
    """Vectorized polars equivalent of `_is_outright(symbol, root)`.

    The previous implementation used `.map_elements(...)` (a row-by-row
    Python callback) to filter the `symbol` column, which is extremely slow
    and memory-heavy on the multi-year per-contract minute-bar files (tens of
    millions of rows). This is the same predicate expressed with native
    (vectorized) polars string expressions.
    """
    symbol = pl.col("symbol")
    suffix = symbol.str.slice(len(root))
    return (
        symbol.str.starts_with(root)
        & ~symbol.str.contains("-", literal=True)
        & ~symbol.str.contains(" ", literal=True)
        & (suffix.str.len_chars() >= 2)
        & suffix.str.slice(0, 1).is_in(list(_MONTH_CODES))
        & suffix.str.slice(1).str.contains(r"^\d+$")
    )


class ContinuousContractDataLoader:
    def _year_daily_symbol_volume(self, root: str, year: int) -> pl.DataFrame:
        """Daily (date, symbol, vol) table for `root` restricted to `year`.

        Cached per (per_contract_1min_dir(), root, year) -- the cache key
        includes the storage dir (not root alone) so it does not go stale
        across a monkeypatched `get_local_storage_dir` (e.g. between unit
        tests that each point at a different tmp_path). Reads and filters
        the raw per-contract minute files for this ONE year, aggregates to
        daily volume, and discards the (large) raw minute data -- only the
        small daily aggregate is retained in the cache.
        """
        pcm_root = per_contract_1min_dir()
        cache_key = (str(pcm_root), root, year)
        if cache_key in _YEAR_DAILY_VOLUME_CACHE:
            return _YEAR_DAILY_VOLUME_CACHE[cache_key]

        empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "vol": pl.UInt64})
        year_dir = pcm_root / f"year={year}"
        if not year_dir.exists():
            _YEAR_DAILY_VOLUME_CACHE[cache_key] = empty
            return empty

        files: list[Path] = sorted(year_dir.rglob("data.parquet"))
        if not files:
            _YEAR_DAILY_VOLUME_CACHE[cache_key] = empty
            return empty

        df = pl.concat([pl.read_parquet(f) for f in files])
        df = df.filter(_outright_filter_expr(root))
        if df.is_empty():
            _YEAR_DAILY_VOLUME_CACHE[cache_key] = empty
            return empty

        daily = df.group_by([
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("symbol"),
        ]).agg(pl.col("volume").sum().alias("vol"))
        _YEAR_DAILY_VOLUME_CACHE[cache_key] = daily
        return daily

    def _active_contract_per_day(
        self, root: str, start: date, end: date
    ) -> pl.DataFrame:
        """Return DataFrame with columns [date, active] for each trading day
        in [start, end] where `active` is the highest-volume outright contract
        of `root` on that day."""
        empty = pl.DataFrame(schema={"date": pl.Date, "active": pl.String})
        parts = [
            self._year_daily_symbol_volume(root, y)
            for y in range(start.year, end.year + 1)
        ]
        parts = [p for p in parts if not p.is_empty()]
        if not parts:
            return empty

        daily = pl.concat(parts) if len(parts) > 1 else parts[0]
        daily = daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        if daily.is_empty():
            return empty

        # For each date, take the symbol with max vol. "symbol" is a
        # deterministic tie-break: on roll days two outright contracts can
        # carry near-equal volume, and without a full sort key `.first()`
        # after a tied sort is order-dependent on the (multi-threaded,
        # non-stable) parquet read/concat order -- observed to flip which
        # contract "wins" a tie between process runs, producing a different
        # roll-date list each time and an intermittent KeyError downstream
        # in `load()`. Sorting on the full (date, vol, symbol) key makes the
        # choice reproducible.
        ranked = daily.sort(["date", "vol", "symbol"], descending=[False, True, False])
        active = ranked.group_by("date").agg(pl.col("symbol").first().alias("active")).sort("date")
        return active

    def detect_roll_dates(
        self, root: str, start: date, end: date
    ) -> list[date]:
        """Return list of dates where the active contract changes.

        Each returned date is the FIRST day the new contract is active.
        """
        active = self._active_contract_per_day(root, start, end)
        if active.is_empty():
            return []
        rolls: list[date] = []
        prev = None
        for row in active.iter_rows(named=True):
            if prev is not None and row["active"] != prev:
                rolls.append(row["date"])
            prev = row["active"]
        return rolls

    def load(
        self,
        root: str,
        method: str = "ratio_adjusted",
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Load continuous contract bars.

        method: "raw" | "ratio_adjusted" | "panama_adjusted"
        start/end: optional date range filter on the output
        """
        if method not in ("raw", "ratio_adjusted", "panama_adjusted"):
            raise ValueError(f"unknown method: {method}")

        sym_dir = continuous_1min_dir() / f"symbol={root}"
        files = sorted(sym_dir.rglob("data.parquet"))
        if not files:
            return pl.DataFrame()
        df = pl.concat([pl.read_parquet(f) for f in files]).sort("timestamp")
        if start is not None:
            df = df.filter(pl.col("timestamp").dt.date() >= start)
        if end is not None:
            df = df.filter(pl.col("timestamp").dt.date() <= end)

        if method == "raw":
            return df

        # Both ratio_adjusted and panama_adjusted need roll dates and per-date close
        if df.is_empty():
            return df
        data_start = df["timestamp"].min().date()
        data_end = df["timestamp"].max().date()
        rolls = self.detect_roll_dates(root, data_start, data_end)
        if not rolls:
            return df

        # Per-date last close for ratio/diff computation at each roll
        daily_close = df.group_by(
            pl.col("timestamp").dt.date().alias("d"),
        ).agg(pl.col("close").last().alias("c")).sort("d")
        close_map = {row["d"]: row["c"] for row in daily_close.iter_rows(named=True)}

        if method == "ratio_adjusted":
            # Walk rolls in reverse, accumulating a ratio factor that applies to
            # all dates strictly before that roll.
            date_factor: dict[date, float] = {d: 1.0 for d in close_map}
            cumulative = 1.0
            for roll_date in reversed(rolls):
                prev_dates = [d for d in close_map if d < roll_date]
                if not prev_dates:
                    continue
                day_before = max(prev_dates)
                old_c = close_map[day_before]
                # The roll date may have no bar in this series (holiday / data
                # gap): the calendar's roll dates and the price series' dates can
                # differ. Snap to the nearest available trading day on-or-after
                # the roll (the new front's first close) instead of a hard
                # lookup that KeyErrors and silently drops the whole root.
                if roll_date in close_map:
                    new_c = close_map[roll_date]
                else:
                    on_or_after = [d for d in close_map if d >= roll_date]
                    if not on_or_after:
                        continue
                    new_c = close_map[min(on_or_after)]
                if old_c == 0:
                    continue
                this_ratio = new_c / old_c
                cumulative *= this_ratio
                for d in [d for d in date_factor if d < roll_date]:
                    date_factor[d] = cumulative

            df_dates = df.with_columns(pl.col("timestamp").dt.date().alias("d"))
            factor_df = pl.DataFrame({
                "d": list(date_factor.keys()),
                "factor": list(date_factor.values()),
            })
            df_adj = df_dates.join(factor_df, on="d", how="left").with_columns([
                (pl.col("open") * pl.col("factor")).alias("open"),
                (pl.col("high") * pl.col("factor")).alias("high"),
                (pl.col("low") * pl.col("factor")).alias("low"),
                (pl.col("close") * pl.col("factor")).alias("close"),
            ]).drop(["d", "factor"])
            return df_adj.sort("timestamp")

        if method == "panama_adjusted":
            date_offset: dict[date, float] = {d: 0.0 for d in close_map}
            cumulative = 0.0
            for roll_date in reversed(rolls):
                prev_dates = [d for d in close_map if d < roll_date]
                if not prev_dates:
                    continue
                day_before = max(prev_dates)
                old_c = close_map[day_before]
                new_c = close_map[roll_date]
                this_diff = new_c - old_c
                cumulative += this_diff
                for d in [d for d in date_offset if d < roll_date]:
                    date_offset[d] = cumulative

            df_dates = df.with_columns(pl.col("timestamp").dt.date().alias("d"))
            offset_df = pl.DataFrame({
                "d": list(date_offset.keys()),
                "offset": list(date_offset.values()),
            })
            df_adj = df_dates.join(offset_df, on="d", how="left").with_columns([
                (pl.col("open") + pl.col("offset")).alias("open"),
                (pl.col("high") + pl.col("offset")).alias("high"),
                (pl.col("low") + pl.col("offset")).alias("low"),
                (pl.col("close") + pl.col("offset")).alias("close"),
            ]).drop(["d", "offset"])
            return df_adj.sort("timestamp")

        raise ValueError(f"unreachable: {method}")

    def ratio_adjust_daily(
        self,
        daily_raw: pl.DataFrame,
        root: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Apply the ratio roll-adjustment directly to a RAW daily series.

        Result-identical to `load(root, "ratio_adjusted")` aggregated to
        daily via `aggregate_to_daily`: the per-date ratio factor is uniform
        within a date, so `last(1min_close * factor) == raw_daily_close *
        factor`. This lets the daily-panel cache path skip re-reading and
        re-aggregating the (much larger) 1-min continuous series.
        """
        df = daily_raw
        if start is not None:
            df = df.filter(pl.col("timestamp").dt.date() >= start)
        if end is not None:
            df = df.filter(pl.col("timestamp").dt.date() <= end)
        if df.is_empty():
            return df

        data_start = df["timestamp"].min().date()
        data_end = df["timestamp"].max().date()
        rolls = self.detect_roll_dates(root, data_start, data_end)
        if not rolls:
            return df

        # Each date already has exactly one row in a daily series, so its
        # close IS the per-date last close (same close_map as the 1-min path).
        close_map = {
            row["timestamp"].date(): row["close"] for row in df.iter_rows(named=True)
        }

        date_factor: dict[date, float] = {d: 1.0 for d in close_map}
        cumulative = 1.0
        for roll_date in reversed(rolls):
            prev_dates = [d for d in close_map if d < roll_date]
            if not prev_dates:
                continue
            day_before = max(prev_dates)
            old_c = close_map[day_before]
            if roll_date in close_map:
                new_c = close_map[roll_date]
            else:
                on_or_after = [d for d in close_map if d >= roll_date]
                if not on_or_after:
                    continue
                new_c = close_map[min(on_or_after)]
            if old_c == 0:
                continue
            this_ratio = new_c / old_c
            cumulative *= this_ratio
            for d in [d for d in date_factor if d < roll_date]:
                date_factor[d] = cumulative

        df_dates = df.with_columns(pl.col("timestamp").dt.date().alias("d"))
        factor_df = pl.DataFrame({
            "d": list(date_factor.keys()),
            "factor": list(date_factor.values()),
        })
        df_adj = df_dates.join(factor_df, on="d", how="left").with_columns([
            (pl.col("open") * pl.col("factor")).alias("open"),
            (pl.col("high") * pl.col("factor")).alias("high"),
            (pl.col("low") * pl.col("factor")).alias("low"),
            (pl.col("close") * pl.col("factor")).alias("close"),
        ]).drop(["d", "factor"])
        return df_adj.sort("timestamp")

    def aggregate_to_daily(
        self, root: str, method: str = "ratio_adjusted",
        start: date | None = None, end: date | None = None,
    ) -> pl.DataFrame:
        """Aggregate minute bars to daily OHLCV."""
        df = self.load(root, method=method, start=start, end=end)
        if df.is_empty():
            return df
        return df.group_by_dynamic(
            "timestamp", every="1d", closed="left", label="left",
        ).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ]).sort("timestamp")

    def aggregate_to_hourly(
        self, root: str, method: str = "ratio_adjusted",
        start: date | None = None, end: date | None = None,
    ) -> pl.DataFrame:
        """Aggregate minute bars to hourly OHLCV."""
        df = self.load(root, method=method, start=start, end=end)
        if df.is_empty():
            return df
        return df.group_by_dynamic(
            "timestamp", every="1h", closed="left", label="left",
        ).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ]).sort("timestamp")
