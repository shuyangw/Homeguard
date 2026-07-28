"""Materialize `regime_state_daily` -- a point-in-time replay of the production
5-state market regime detector.

`MarketRegimeDetector.analyze_regime_history()` is a verified causal replay
(it truncates both input frames to `<= date` before each classification and
the VIX percentile uses a trailing 252-row window of the already-truncated
frame), but it has NO persistence and is O(n^2) in dates, with several
downstream consumers each paying that cost. This module runs it ONCE against
the MATERIALIZED VIX spot snapshot (`alt_data/vix/vix_spot.parquet`) plus SPY
daily bars, and writes the result with provenance.

`replay_regime_history` reproduces `analyze_regime_history` exactly -- same
loop, same truncation -- and additionally captures the detector's
`last_indicators` and `last_regime_scores`, which the upstream method
discards. Equivalence is asserted in tests/data/test_vix_spot_and_regime.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data.acquisition.plugins.vix_spot import (
    REQUIRED_END,
    REQUIRED_START,
    git_sha,
    load_vix_spot,
    load_vix_spot_meta,
)
from src.settings import get_local_storage_dir
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger(__name__)

VALID_REGIMES = {"STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "UNPREDICTABLE", "BEAR"}

SPY_SYMBOL = "SPY"
SPY_FETCH_START = "2011-01-01"

DATA_VINTAGE_CAVEAT = (
    "Replay uses TODAY's SPY/VIX series. If that data was revised or backfilled "
    "since, the recomputed state differs from what was known at the time. This is "
    "stated on results, not assumed away."
)


def _regime_dir() -> Path:
    return get_local_storage_dir() / "alt_data" / "regime"


REGIME_STATE_PATH = _regime_dir() / "regime_state_daily.parquet"
REGIME_STATE_META_PATH = _regime_dir() / "regime_state_daily.meta.json"


def replay_regime_history(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Causal replay -- identical semantics to
    MarketRegimeDetector.analyze_regime_history, plus indicator/score columns.
    """
    from src.strategies.advanced.market_regime_detector import MarketRegimeDetector

    detector = MarketRegimeDetector()
    mask = (spy_data.index >= start_date) & (spy_data.index <= end_date)
    analysis_dates = spy_data[mask].index

    rows = []
    for date in analysis_dates:
        spy_subset = spy_data[spy_data.index <= date]
        vix_subset = vix_data[vix_data.index <= date]
        regime, confidence = detector.classify_regime(spy_subset, vix_subset, date)

        row = {
            "date": date,
            "regime": regime,
            "confidence": confidence,
            "spy_close": float(spy_subset["close"].iloc[-1]),
            "vix": float(vix_subset["close"].iloc[-1]),
        }
        for key, value in (detector.last_indicators or {}).items():
            if key == "vix":
                continue
            row[key] = value
        for regime_name, score in (detector.last_regime_scores or {}).items():
            row[f"score_{regime_name}"] = float(score)
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def _load_spy_daily(start: str, end: str) -> Tuple[pd.DataFrame, str]:
    """Load SPY dailies through the repo's normal provider chain."""
    from src.data.providers.factory import create_data_provider

    provider = create_data_provider()
    df = provider.get_historical_bars(
        SPY_SYMBOL,
        datetime.fromisoformat(start),
        datetime.fromisoformat(end),
        timeframe="1D",
    )
    if df is None or df.empty:
        raise RuntimeError("regime_state: SPY daily bars unavailable from provider chain")
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx.normalize(), name="date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df, type(provider).__name__


def build(
    start: str = REQUIRED_START,
    end: Optional[str] = None,
    status=None,
) -> Path:
    end = end or str(pd.Timestamp(tz.now().date()).date())

    vix = load_vix_spot()
    vix_meta = load_vix_spot_meta()
    spy, spy_source = _load_spy_daily(SPY_FETCH_START, end)
    if status is not None:
        status.heartbeat(note=f"inputs loaded spy={len(spy)} vix={len(vix)}")

    logger.info(
        f"[regime_state] SPY {len(spy)} rows {spy.index.min().date()} -> {spy.index.max().date()} "
        f"via {spy_source}; VIX spot {len(vix)} rows from {vix_meta['source']}"
    )

    # Never replay past the last date where BOTH inputs have a completed
    # session. Today's bar is partial while the market is open, and a partial
    # SPY bar paired with a stale VIX close would be a fabricated state.
    today = pd.Timestamp(tz.now().date())
    spy = spy[spy.index < today]
    vix_frame = vix[vix.index < today]
    effective_end = min(pd.Timestamp(end), spy.index.max(), vix_frame.index.max())
    if effective_end < pd.Timestamp(end):
        logger.info(f"[regime_state] end clamped {end} -> {effective_end.date()} (last complete session in BOTH inputs)")
    vix = vix_frame

    t0 = tz.now()
    history = replay_regime_history(spy, vix, pd.Timestamp(start), effective_end)
    runtime_s = (tz.now() - t0).total_seconds()
    if status is not None:
        status.heartbeat(note=f"replay done rows={len(history)} secs={runtime_s:.0f}")

    bad = set(history["regime"].unique()) - VALID_REGIMES
    if bad:
        raise ValueError(f"regime_state: unexpected regime labels {bad}")

    out_dir = _regime_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = REGIME_STATE_PATH.with_suffix(".parquet.tmp")
    history.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, REGIME_STATE_PATH)

    counts = history["regime"].value_counts()
    distribution = {
        r: {"count": int(counts.get(r, 0)),
            "pct": round(float(counts.get(r, 0)) / len(history) * 100.0, 2)}
        for r in sorted(VALID_REGIMES)
    }

    meta = {
        "dataset": "regime_state_daily",
        "detector": "src/strategies/advanced/market_regime_detector.py::MarketRegimeDetector",
        "detector_git_sha": git_sha(),
        "snapshot_timestamp": tz.now().isoformat(),
        "vix_spot_source": vix_meta["source"],
        "vix_spot_snapshot_timestamp": vix_meta["snapshot_timestamp"],
        "vix_spot_path": str(REGIME_STATE_PATH.parent.parent / "vix" / "vix_spot.parquet"),
        "spy_source": spy_source,
        "spy_symbol": SPY_SYMBOL,
        "spy_as_of": str(spy.index.max().date()),
        "spy_rows": int(len(spy)),
        "rows": int(len(history)),
        "date_min": str(history.index.min().date()),
        "date_max": str(history.index.max().date()),
        "registered_window": {"start": REQUIRED_START, "end": REQUIRED_END},
        "replay_runtime_seconds": round(runtime_s, 1),
        "regime_distribution": distribution,
        "columns": list(history.reset_index().columns),
        "data_vintage_caveat": DATA_VINTAGE_CAVEAT,
    }
    REGIME_STATE_META_PATH.write_text(json.dumps(meta, indent=2))

    logger.info(
        f"[regime_state] wrote {len(history)} rows {meta['date_min']} -> {meta['date_max']} "
        f"to {REGIME_STATE_PATH} ({runtime_s:.0f}s)"
    )
    for name, d in distribution.items():
        logger.info(f"[regime_state]   {name:<14} {d['count']:>5}  {d['pct']:>6.2f}%")
    logger.warning(f"[regime_state] [!] {DATA_VINTAGE_CAVEAT}")
    return REGIME_STATE_PATH


def load_regime_state_daily() -> pd.DataFrame:
    if not REGIME_STATE_PATH.exists():
        raise FileNotFoundError(
            f"regime_state_daily not materialized at {REGIME_STATE_PATH} -- "
            f"run scripts/data/build_regime_state_daily.py"
        )
    df = pd.read_parquet(REGIME_STATE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_regime_state_meta() -> dict:
    return json.loads(REGIME_STATE_META_PATH.read_text())
