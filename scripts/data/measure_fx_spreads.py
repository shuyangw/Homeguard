"""Measure EMPIRICAL FX spreads from Dukascopy bid/ask quotes.

Why: the FX cost model has always used ASSUMED spreads (a per-tier pip constant,
plus a synthetic hour-of-week shape in the `spread_model` artifact). Nothing was
ever measured. This samples real bid/ask at 1-minute resolution and writes an
empirical table so the cost model can be calibrated to observed reality rather
than to a guess.

Sampling, not full history: one month per sample year per pair. A month gives
~780 observations per hour-of-week bucket, which is ample for a median, while a
full 15-year pull for every pair would run for hours. Sample years are spread
across liquidity regimes so a structural change in spreads is visible.

IMPORTANT interpretation caveat, recorded here so it travels with the data:
Dukascopy is an ECN-style feed and quotes RAW spreads; a retail account also pays
commission. These numbers are therefore a LOWER BOUND on realistic retail cost,
not a drop-in replacement for the taker cost model. Treat them as the floor.

Usage: PYTHONPATH=$(pwd) python scripts/data/measure_fx_spreads.py
"""
from __future__ import annotations

import pandas as pd

from src.data.acquisition.plugins.dukascopy_fx import fetch_pair_month_quotes
from src.settings import get_local_storage_dir
from src.utils import logger

# Pairs spanning the liquidity spectrum, from tightest major to widest EM.
PAIRS = [
    # G10 majors and crosses that Dukascopy quotes directly. The three derived
    # Nordic crosses (NOKSEK/NOKJPY/SEKJPY) are triangulated, not quoted, so they
    # have no measurable spread and are excluded.
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "EURCHF", "CHFJPY", "AUDJPY", "NZDJPY", "AUDNZD",
    "EURNOK", "EURSEK", "USDNOK", "USDSEK",
    "XAUUSD", "XAGUSD",
    # EM
    "USDMXN", "USDZAR", "USDTRY", "USDPLN", "USDHUF", "USDCNH",
]
SAMPLE_MONTHS = [(2015, 6), (2020, 6), (2024, 6)]


def measure(pairs=None, months=None) -> pd.DataFrame:
    pairs = pairs or PAIRS
    months = months or SAMPLE_MONTHS
    rows = []
    for pair in pairs:
        for year, month in months:
            try:
                q = fetch_pair_month_quotes(pair, year, month)
            except Exception as e:
                logger.error(f"[spreads] {pair} {year}-{month:02d} failed: {e}")
                continue
            if q.empty:
                logger.warning(f"[spreads] {pair} {year}-{month:02d} returned no quotes")
                continue
            s = q["spread_pips"]
            idx = pd.DatetimeIndex(q.index)
            # hour-of-week: Monday 00:00 UTC = 0 .. Sunday 23:00 UTC = 167
            how = idx.dayofweek * 24 + idx.hour
            for h, grp in s.groupby(how):
                rows.append({"pair": pair, "year": year, "month": month,
                             "hour_of_week": int(h), "n": int(grp.size),
                             "spread_p50": float(grp.median()),
                             "spread_p90": float(grp.quantile(0.90))})
            logger.info(f"[spreads] {pair} {year}-{month:02d}: {len(s)} quotes, "
                        f"median {s.median():.3f} pips")
    return pd.DataFrame(rows)


def main() -> None:
    df = measure()
    if df.empty:
        logger.error("[spreads] no data measured")
        return
    out_dir = get_local_storage_dir() / "artifacts" / "fx" / "measured_spreads"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "table.parquet"
    df.to_parquet(path, index=False)
    logger.success(f"[spreads] wrote {path}: {len(df)} rows, {df['pair'].nunique()} pairs")

    summary = (df.groupby(["pair", "year"])
                 .apply(lambda g: pd.Series({
                     "median_spread_pips": (g["spread_p50"] * g["n"]).sum() / g["n"].sum(),
                     "n_quotes": int(g["n"].sum())}), include_groups=False)
                 .reset_index())
    logger.info("[spreads] measured median spread (pips) by pair/year:\n"
                + summary.to_string(index=False))


if __name__ == "__main__":
    main()
