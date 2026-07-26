"""Build the MEASURED hour-of-week FX spread surface as a committed multiplier table.

Why this exists: the intraday cost path (`fx_round_trip_pips`) charges ONE
constant per run -- a tier midpoint times a hand-set session multiplier. Real
spreads vary far more than that: EURUSD ranges 0.30 to 10.20 pips across the
hour-of-week (34x), USDJPY 0.30 to 3.80 (12.7x). An intraday strategy
concentrates its trades in specific hours, so a flat spread is wrong in
whichever direction that strategy happens to trade.

Shape, not level. The table holds a MULTIPLIER on the per-pair round-trip level
already baked into `_MEASURED_RT_BPS`, normalised so the quote-weighted mean
multiplier is 1.0. Two reasons: the level keeps a single source of truth, and a
ratio of spreads is unit-free, so the pip-denominated sample needs no
pips-to-bps price conversion to contribute its shape.

Sources, best-available per pair:
  - 5 majors with local tick-derived quotes: full history, direct from
    bid/ask (`fx/massive/quotes_minute_aggregated`).
  - 20 further pairs: the existing Dukascopy 3-month-per-year sample
    (`artifacts/fx/measured_spreads/table.parquet`, written by
    scripts/data/measure_fx_spreads.py).

Hours with no quotes (the weekend close) are absent from the table by design;
the consumer treats them as the pair's widest observed hour rather than
inventing a value.

Usage: PYTHONPATH=$(pwd) python scripts/data/build_fx_hour_of_week_cost.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

OUT_PATH = Path("config/costs/fx_hour_of_week_spread.csv")
_QUOTES_SUBDIR = ("fx", "massive", "quotes_minute_aggregated")
_SAMPLE_TABLE = ("artifacts", "fx", "measured_spreads", "table.parquet")
_MIN_QUOTES_PER_HOUR = 100


def _hour_of_week(idx: pd.DatetimeIndex) -> pd.Series:
    """Monday 00:00 UTC = 0 ... Sunday 23:00 UTC = 167."""
    return pd.Series(idx.dayofweek * 24 + idx.hour, index=idx)


def _normalise(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Quote-weighted normalisation so the mean multiplier is exactly 1.0."""
    weighted_mean = (df[value_col] * df["n_quotes"]).sum() / df["n_quotes"].sum()
    out = df.copy()
    out["spread_multiplier"] = (out[value_col] / weighted_mean).round(4)
    return out[["hour_of_week", "spread_multiplier", "n_quotes"]]


def _from_local_quotes(pair: str) -> pd.DataFrame | None:
    root = get_local_storage_dir().joinpath(*_QUOTES_SUBDIR) / f"symbol={pair}"
    files = sorted(root.glob("**/*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(f, columns=["timestamp", "spread_p50",
                                          "bid_close", "ask_close"]) for f in files]
    q = pd.concat(frames, ignore_index=True)
    idx = pd.DatetimeIndex(q["timestamp"])
    mid = (q["bid_close"] + q["ask_close"]) / 2.0
    q = q.assign(hour_of_week=(idx.dayofweek * 24 + idx.hour).values,
                 spread_bps=(q["spread_p50"] / mid * 1e4).values)
    grp = q.groupby("hour_of_week").agg(spread=("spread_bps", "median"),
                                        n_quotes=("spread_bps", "size")).reset_index()
    grp = grp[grp["n_quotes"] >= _MIN_QUOTES_PER_HOUR]
    return _normalise(grp, "spread")


def _from_sample_table(sample: pd.DataFrame, pair: str) -> pd.DataFrame | None:
    rows = sample[sample["pair"] == pair]
    if rows.empty:
        return None
    grp = (rows.groupby("hour_of_week")
               .apply(lambda g: pd.Series({
                   "spread": (g["spread_p50"] * g["n"]).sum() / g["n"].sum(),
                   "n_quotes": int(g["n"].sum())}), include_groups=False)
               .reset_index())
    grp = grp[grp["n_quotes"] >= _MIN_QUOTES_PER_HOUR]
    if grp.empty:
        return None
    return _normalise(grp, "spread")


def build() -> pd.DataFrame:
    sample_path = get_local_storage_dir().joinpath(*_SAMPLE_TABLE)
    sample = pd.read_parquet(sample_path) if sample_path.exists() else pd.DataFrame()
    if sample.empty:
        logger.warning(f"[how_cost] no sample table at {sample_path}; "
                       "only locally-quoted pairs will be measured")

    quote_root = get_local_storage_dir().joinpath(*_QUOTES_SUBDIR)
    quoted = sorted(p.name.replace("symbol=", "") for p in quote_root.glob("symbol=*"))
    sampled = sorted(sample["pair"].unique()) if not sample.empty else []

    out = []
    for pair in sorted(set(quoted) | set(sampled)):
        table = _from_local_quotes(pair) if pair in quoted else None
        source = "local_quotes_full_history"
        if table is None or table.empty:
            table = _from_sample_table(sample, pair)
            source = "dukascopy_sample"
        if table is None or table.empty:
            logger.warning(f"[how_cost] {pair}: no usable measurement, skipped")
            continue
        spread = table["spread_multiplier"]
        logger.info(f"[how_cost] {pair:7s} {source:26s} hours={len(table):3d} "
                    f"mult min {spread.min():.2f} max {spread.max():.2f} "
                    f"({spread.max() / spread.min():.1f}x)")
        out.append(table.assign(pair=pair, source=source))

    if not out:
        return pd.DataFrame()
    return (pd.concat(out, ignore_index=True)
              [["pair", "hour_of_week", "spread_multiplier", "n_quotes", "source"]]
              .sort_values(["pair", "hour_of_week"], ignore_index=True))


def main() -> None:
    df = build()
    if df.empty:
        logger.error("[how_cost] nothing measured; table not written")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.success(f"[how_cost] wrote {OUT_PATH}: {len(df)} rows, "
                   f"{df['pair'].nunique()} pairs")


if __name__ == "__main__":
    main()
