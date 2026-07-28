"""Reproducibility tests for the materialized VIX-spot and regime-state datasets.

These datasets exist so that the RAMP/options research stack stops FETCHING
VIX at call time (a fetched series is not reproducible). The tests below are
the contract those materializations must satisfy:

  - schema / dtype / monotonic-unique date index for the VIX spot series
  - provenance sidecar exists and carries a snapshot timestamp + source
  - regime_state_daily only ever contains the 5 valid regime labels
  - CAUSALITY: the regime stored at date t is exactly what the detector
    produces from inputs truncated to <= t (point-in-time proof)
  - the registered 2012-06 -> 2026-02 window is covered

Every test that needs the artifacts skips (not fails) when they have not been
built yet, so the suite stays green on a fresh clone.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.acquisition.plugins.vix_spot import (
    VIX_SPOT_META_PATH,
    VIX_SPOT_PATH,
    load_vix_spot,
)
from src.data.derivations.regime_state import (
    REGIME_STATE_META_PATH,
    REGIME_STATE_PATH,
    REQUIRED_END,
    REQUIRED_START,
    VALID_REGIMES,
    load_regime_state_daily,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require(path):
    if not path.exists():
        pytest.skip(f"artifact not built: {path}")


# --------------------------------------------------------------------------
# 1. VIX spot schema
# --------------------------------------------------------------------------

def test_vix_spot_schema_and_index():
    _require(VIX_SPOT_PATH)
    df = load_vix_spot()

    for col in ("open", "high", "low", "close"):
        assert col in df.columns, f"missing column: {col}"
        assert pd.api.types.is_float_dtype(df[col]), f"{col} is not float"

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["close"].notna().all()


def test_vix_spot_values_are_plausible():
    _require(VIX_SPOT_PATH)
    df = load_vix_spot()
    assert (df["close"] > 5.0).all()
    assert (df["close"] < 200.0).all()


# --------------------------------------------------------------------------
# 2. Provenance sidecar
# --------------------------------------------------------------------------

def test_vix_spot_meta_has_provenance():
    _require(VIX_SPOT_META_PATH)
    meta = json.loads(VIX_SPOT_META_PATH.read_text())
    assert meta.get("source"), "meta is missing 'source'"
    assert meta.get("snapshot_timestamp"), "meta is missing 'snapshot_timestamp'"
    # must parse as an ISO timestamp
    pd.Timestamp(meta["snapshot_timestamp"])
    assert meta.get("rows", 0) > 0
    assert meta.get("date_min") and meta.get("date_max")


def test_regime_state_meta_has_provenance_and_vintage_caveat():
    _require(REGIME_STATE_META_PATH)
    meta = json.loads(REGIME_STATE_META_PATH.read_text())
    assert meta.get("vix_spot_snapshot_timestamp")
    assert meta.get("spy_source")
    assert meta.get("spy_as_of")
    assert "data_vintage_caveat" in meta
    assert "revised or backfilled" in meta["data_vintage_caveat"]


# --------------------------------------------------------------------------
# 3. Regime label integrity
# --------------------------------------------------------------------------

def test_regime_state_labels_and_nulls():
    _require(REGIME_STATE_PATH)
    df = load_regime_state_daily()
    assert set(df["regime"].unique()).issubset(VALID_REGIMES)
    assert df["regime"].notna().all()
    assert df.index.notna().all()
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["confidence"].between(0.0, 1.0).all()


# --------------------------------------------------------------------------
# 4. CAUSALITY REGRESSION -- the point of the whole exercise
# --------------------------------------------------------------------------

def test_regime_is_point_in_time():
    """Re-running the detector on inputs truncated to <= t must reproduce the
    materialized regime/confidence at t.

    The detector only ever reads spy_data['close'] and vix_data['close'], and
    the materialized table carries both series (spy_close, vix), so this test
    is fully offline -- it replays from the artifact itself.
    """
    _require(REGIME_STATE_PATH)
    from src.strategies.advanced.market_regime_detector import MarketRegimeDetector

    df = load_regime_state_daily()
    spy = pd.DataFrame({"close": df["spy_close"].astype(float)})
    vix = pd.DataFrame({"close": df["vix"].astype(float)})

    # Need >= 252 prior rows for the VIX percentile window; sample from there on.
    positions = range(300, len(df), max(1, (len(df) - 300) // 8))
    detector = MarketRegimeDetector()
    checked = 0
    for pos in positions:
        t = df.index[pos]
        regime, confidence = detector.classify_regime(
            spy.loc[:t], vix.loc[:t], t,
        )
        assert regime == df["regime"].iloc[pos], (
            f"regime mismatch at {t}: replay={regime} stored={df['regime'].iloc[pos]}"
        )
        assert abs(confidence - float(df["confidence"].iloc[pos])) < 1e-9
        checked += 1
    assert checked >= 5


# --------------------------------------------------------------------------
# 5. Coverage of the registered window
# --------------------------------------------------------------------------

def test_vix_spot_covers_registered_window():
    _require(VIX_SPOT_PATH)
    df = load_vix_spot()
    assert df.index.min() <= pd.Timestamp(REQUIRED_START)
    assert df.index.max() >= pd.Timestamp(REQUIRED_END)


def test_regime_state_covers_registered_window():
    _require(REGIME_STATE_PATH)
    df = load_regime_state_daily()
    # first regime row must be at or before the registered start, last at or after end
    assert df.index.min() <= pd.Timestamp(REQUIRED_START) + pd.Timedelta(days=7)
    assert df.index.max() >= pd.Timestamp(REQUIRED_END)


# --------------------------------------------------------------------------
# Driver equivalence: our replay loop must match the detector's own
# analyze_regime_history() on the same inputs (we only add columns).
# --------------------------------------------------------------------------

def test_replay_matches_analyze_regime_history():
    _require(REGIME_STATE_PATH)
    from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
    from src.data.derivations.regime_state import replay_regime_history

    df = load_regime_state_daily()
    spy = pd.DataFrame({"close": df["spy_close"].astype(float)})
    vix = pd.DataFrame({"close": df["vix"].astype(float)})

    start = df.index[400]
    end = df.index[430]
    ours = replay_regime_history(spy, vix, start, end)
    theirs = MarketRegimeDetector().analyze_regime_history(spy, vix, start, end)

    assert list(ours.index) == list(theirs.index)
    assert list(ours["regime"]) == list(theirs["regime"])
    pd.testing.assert_series_equal(
        ours["confidence"], theirs["confidence"], check_names=False,
    )
