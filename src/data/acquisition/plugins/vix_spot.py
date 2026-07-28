"""VIX INDEX SPOT (^VIX) daily OHLC -> materialized, provenance-stamped parquet.

Distinct from `cboe_vix.py`, which materializes the VIX FUTURES term structure
(alt_data/vix/vx_curve.parquet -- vx1/vx2 settlements). The regime detector
(`src/strategies/advanced/market_regime_detector.py`) consumes the VIX INDEX
SPOT close, which until now was FETCHED at call time via
`src/utils/vix_provider.py` (yfinance ^VIX). A fetched series is not
reproducible -- it can be revised, rate-limited, or fail outright. This module
snapshots it once, with provenance, so downstream replays are deterministic.

Source chain (mirrors VIXProvider's, reusing it rather than forking it):
  1. yfinance ^VIX -- full OHLC
  2. FRED VIXCLS via `VIXProvider._fetch_fred` -- close only (open/high/low NaN)

Writes alt_data/vix/vix_spot.parquet + alt_data/vix/vix_spot.meta.json.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

VIX_TICKER = "^VIX"

# Registered coverage window for the equity-options Phase-1 data layer.
REQUIRED_START = "2012-06-01"
REQUIRED_END = "2026-02-28"

# Fetch earlier than REQUIRED_START so the detector's 252-day VIX percentile
# window is warm on the first required date.
FETCH_START = "2011-01-01"

_OHLC = ("open", "high", "low", "close")


def _vix_dir() -> Path:
    return get_local_storage_dir() / "alt_data" / "vix"


VIX_SPOT_PATH = _vix_dir() / "vix_spot.parquet"
VIX_SPOT_META_PATH = _vix_dir() / "vix_spot.meta.json"


def git_sha() -> Optional[str]:
    """Read HEAD's SHA straight off disk -- no git subprocess."""
    try:
        from src.settings import PROJECT_ROOT

        head = Path(PROJECT_ROOT) / ".git" / "HEAD"
        if not head.exists():
            return None
        content = head.read_text().strip()
        if content.startswith("ref:"):
            ref = (Path(PROJECT_ROOT) / ".git" / content.split(" ", 1)[1].strip())
            return ref.read_text().strip() if ref.exists() else None
        return content
    except Exception as exc:
        logger.warning(f"[vix_spot] could not read git SHA: {exc}")
        return None


def _fetch_yfinance(start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf

        raw = yf.download(
            VIX_TICKER, start=start, end=end, progress=False, auto_adjust=False,
        )
    except Exception as exc:
        logger.error(f"[vix_spot] yfinance fetch failed: {exc}")
        return None
    if raw is None or raw.empty:
        logger.error("[vix_spot] yfinance returned empty data")
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    cols = {c.lower(): c for c in raw.columns}
    missing = [c for c in _OHLC if c not in cols]
    if missing:
        logger.error(f"[vix_spot] yfinance response missing columns: {missing}")
        return None
    out = pd.DataFrame({c: raw[cols[c]].astype(float).values for c in _OHLC})
    out.index = pd.DatetimeIndex(pd.to_datetime(raw.index).tz_localize(None).normalize(), name="date")
    return out


def _fetch_fred(start: str, end: str) -> Optional[pd.DataFrame]:
    """Fallback: reuse VIXProvider's FRED path (close only)."""
    try:
        from src.utils.vix_provider import VIXProvider

        df = VIXProvider()._fetch_fred(
            datetime.fromisoformat(start), datetime.fromisoformat(end),
        )
    except Exception as exc:
        logger.error(f"[vix_spot] FRED fallback failed: {exc}")
        return None
    if df is None or df.empty:
        return None
    out = pd.DataFrame({"open": float("nan"), "high": float("nan"),
                        "low": float("nan"), "close": df["close"].astype(float).values})
    out.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).tz_localize(None).normalize(), name="date")
    return out


def fetch_vix_spot(start: str = FETCH_START, end: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Fetch VIX index spot OHLC. Returns (frame, source_name)."""
    # yfinance treats `end` as EXCLUSIVE. Defaulting to today therefore stops at
    # the last COMPLETE session -- a partially-formed live bar must never enter a
    # reproducibility snapshot.
    end = end or tz.now().date().isoformat()
    df = _fetch_yfinance(start, end)
    if df is not None and not df.empty:
        return _clean(df), f"yfinance:{VIX_TICKER}"
    logger.warning("[vix_spot] yfinance unavailable -> falling back to FRED VIXCLS (close only)")
    df = _fetch_fred(start, end)
    if df is not None and not df.empty:
        return _clean(df), "FRED:VIXCLS"
    raise RuntimeError("vix_spot: all sources failed (yfinance ^VIX, FRED VIXCLS)")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["close"].notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def audit_vix_spot(df: pd.DataFrame) -> dict:
    """Report data-quality findings. Reports only -- never repairs."""
    sessions = pd.bdate_range(df.index.min(), df.index.max())
    present = df.index
    missing = sessions.difference(present)
    gaps = []
    if len(missing) > 0:
        run_start = prev = missing[0]
        for d in missing[1:]:
            if (d - prev).days <= 3:  # same weekday run across a weekend
                prev = d
                continue
            gaps.append((run_start, prev))
            run_start = prev = d
        gaps.append((run_start, prev))
    long_gaps = [
        {"start": str(a.date()), "end": str(b.date()),
         "business_days": int(len(pd.bdate_range(a, b)))}
        for a, b in gaps
        if len(pd.bdate_range(a, b)) > 3
    ]
    close = df["close"]
    return {
        "rows": int(len(df)),
        "business_days_in_span": int(len(sessions)),
        "missing_business_days": int(len(missing)),
        "gaps_over_3_business_days": long_gaps,
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "close_mean": float(close.mean()),
        "non_positive_closes": int((close <= 0).sum()),
        "closes_outside_5_200": int(((close <= 5.0) | (close >= 200.0)).sum()),
    }


def build(start: str = FETCH_START, end: Optional[str] = None) -> Path:
    df, source = fetch_vix_spot(start, end)
    audit = audit_vix_spot(df)

    out_dir = _vix_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = VIX_SPOT_PATH.with_suffix(".parquet.tmp")
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, VIX_SPOT_PATH)

    meta = {
        "dataset": "vix_spot",
        "description": "CBOE VIX INDEX SPOT daily OHLC (not VIX futures)",
        "source": source,
        "snapshot_timestamp": tz.now().isoformat(),
        "git_sha": git_sha(),
        "rows": int(len(df)),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "registered_window": {"start": REQUIRED_START, "end": REQUIRED_END},
        "audit": audit,
    }
    VIX_SPOT_META_PATH.write_text(json.dumps(meta, indent=2))

    logger.info(
        f"[vix_spot] wrote {len(df)} rows {meta['date_min']} -> {meta['date_max']} "
        f"from {source} to {VIX_SPOT_PATH}"
    )
    logger.info(
        f"[vix_spot] audit: close min/mean/max = {audit['close_min']:.2f}/"
        f"{audit['close_mean']:.2f}/{audit['close_max']:.2f}, "
        f"missing business days = {audit['missing_business_days']}, "
        f"gaps > 3 bd = {len(audit['gaps_over_3_business_days'])}, "
        f"closes outside (5,200) = {audit['closes_outside_5_200']}"
    )
    for g in audit["gaps_over_3_business_days"]:
        logger.warning(f"[vix_spot] [!] gap {g['start']} -> {g['end']} ({g['business_days']} bd) -- NOT filled")
    return VIX_SPOT_PATH


def load_vix_spot() -> pd.DataFrame:
    """Load the materialized VIX spot series, indexed by tz-naive date."""
    if not VIX_SPOT_PATH.exists():
        raise FileNotFoundError(
            f"vix_spot not materialized at {VIX_SPOT_PATH} -- "
            f"run scripts/data/build_vix_spot.py"
        )
    df = pd.read_parquet(VIX_SPOT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_vix_spot_meta() -> dict:
    return json.loads(VIX_SPOT_META_PATH.read_text())
