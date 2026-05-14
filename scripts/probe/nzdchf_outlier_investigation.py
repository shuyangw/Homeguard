"""Phase 0 Probe 3: NZDCHF triangulation outlier investigation.

Tests three hypotheses on why NZDCHF triangulation (vs NZDUSD * USDCHF)
shows 0.99% outliers > 50 bps and 11.33 bps std:

1. Stale-bar lag (NZDCHF quotes stale vs components)
2. Bad-tick noise (one-off mis-prints)
3. Asia-session microstructure thinness

Reads existing fx_1min/symbol={NZDCHF,NZDUSD,USDCHF}/year=2025/month=12/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
logger = get_logger(__name__)


def load_pair(sym: str) -> pl.DataFrame:
    root = get_local_storage_dir()
    f = root / "fx_1min" / f"symbol={sym}" / "year=2025" / "month=12" / "data.parquet"
    return pl.read_parquet(f).select(
        "timestamp", pl.col("close").alias(sym.lower()),
    )


def baseline_outliers() -> dict:
    """Reproduce the prior 0.99% finding."""
    nzdchf = load_pair("NZDCHF")
    nzdusd = load_pair("NZDUSD")
    usdchf = load_pair("USDCHF")
    j = nzdusd.join(usdchf, on="timestamp").join(nzdchf, on="timestamp")
    j = j.with_columns(
        ((pl.col("nzdusd") * pl.col("usdchf") - pl.col("nzdchf"))
         / pl.col("nzdchf") * 10000).alias("bps"),
    )
    bps = j["bps"]
    return {
        "n_bars": j.height,
        "mean_bps": float(bps.mean()),
        "std_bps": float(bps.std()),
        "outliers_pct": float((bps.abs() > 50).mean()) * 100,
        "df": j,
    }


def lag_adjusted(j: pl.DataFrame, lag_minutes: int) -> dict:
    """Triangulate with NZDCHF shifted by lag_minutes vs components."""
    shifted = j.with_columns(
        pl.col("nzdchf").shift(lag_minutes).alias("nzdchf_lag"),
    ).drop_nulls()
    shifted = shifted.with_columns(
        ((pl.col("nzdusd") * pl.col("usdchf") - pl.col("nzdchf_lag"))
         / pl.col("nzdchf_lag") * 10000).alias("bps_lag"),
    )
    bps = shifted["bps_lag"]
    return {
        "lag_minutes": lag_minutes,
        "n_bars": shifted.height,
        "mean_bps": float(bps.mean()),
        "std_bps": float(bps.std()),
        "outliers_pct": float((bps.abs() > 50).mean()) * 100,
    }


def mad_filtered(j: pl.DataFrame, window: int = 60, threshold: float = 6.0) -> dict:
    """Filter ticks where |price - rolling_median| / mad > threshold; re-triangulate."""
    out = j.clone()
    for col in ["nzdchf", "nzdusd", "usdchf"]:
        rolling_med = out[col].rolling_median(window_size=window)
        rolling_mad = (out[col] - rolling_med).abs().rolling_median(window_size=window)
        deviation = (out[col] - rolling_med).abs() / rolling_mad.fill_null(1e-9)
        out = out.with_columns(
            pl.when(deviation > threshold).then(None).otherwise(out[col]).alias(col),
        )
    out = out.drop_nulls()
    out = out.with_columns(
        ((pl.col("nzdusd") * pl.col("usdchf") - pl.col("nzdchf"))
         / pl.col("nzdchf") * 10000).alias("bps_mad"),
    )
    bps = out["bps_mad"]
    return {
        "kept_bars": out.height,
        "dropped_pct": (1 - out.height / j.height) * 100,
        "mean_bps": float(bps.mean()),
        "std_bps": float(bps.std()),
        "outliers_pct": float((bps.abs() > 50).mean()) * 100,
    }


def hour_breakdown(j: pl.DataFrame) -> pl.DataFrame:
    """Outlier rate per UTC hour."""
    h = j.with_columns(
        pl.col("timestamp").dt.hour().alias("hr"),
        ((pl.col("nzdusd") * pl.col("usdchf") - pl.col("nzdchf"))
         / pl.col("nzdchf") * 10000).alias("bps"),
    )
    return h.group_by("hr").agg(
        pl.len().alias("n"),
        pl.col("bps").abs().mean().alias("mean_abs_bps"),
        (pl.col("bps").abs() > 50).mean().alias("outlier_rate"),
    ).sort("hr")


def main() -> int:
    base = baseline_outliers()
    print(f"BASELINE: n={base['n_bars']:,}, mean={base['mean_bps']:.2f}bps, "
          f"std={base['std_bps']:.2f}bps, outliers={base['outliers_pct']:.2f}%")

    print("\nLAG SWEEP (NZDCHF shift relative to NZDUSD * USDCHF):")
    for lag in [-3, -2, -1, 0, 1, 2, 3]:
        r = lag_adjusted(base["df"], lag)
        print(f"  lag={lag:+d}: std={r['std_bps']:.2f}bps, outliers={r['outliers_pct']:.2f}%")

    print("\nMAD FILTER (60-min window, 6x threshold):")
    r = mad_filtered(base["df"])
    print(f"  dropped={r['dropped_pct']:.2f}%, std={r['std_bps']:.2f}bps, "
          f"outliers={r['outliers_pct']:.2f}%")

    print("\nUTC HOUR BREAKDOWN:")
    print(hour_breakdown(base["df"]))

    print("\nRecommendation: see verdict in Phase 0 results doc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
