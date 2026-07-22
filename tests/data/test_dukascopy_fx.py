import pandas as pd

from src.data.acquisition.plugins.dukascopy_fx import (
    mid_ohlc, derive_ohlc, to_canonical, DUKA_MAP,
)
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA


def _ohlc(idx, base):
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base, "volume": [1.0] * len(idx)},
        index=idx,
    )


def test_mid_ohlc_averages_bid_ask():
    idx = pd.date_range("2020-11-16", periods=2, freq="min", tz="UTC")
    bid = _ohlc(idx, [1.00, 1.02])
    ask = _ohlc(idx, [1.02, 1.06])
    mid = mid_ohlc(bid, ask)
    assert list(mid["close"]) == [1.01, 1.04]


def test_mid_ohlc_only_shared_timestamps():
    a = pd.date_range("2020-11-16 00:00", periods=3, freq="min", tz="UTC")
    b = pd.date_range("2020-11-16 00:01", periods=3, freq="min", tz="UTC")
    mid = mid_ohlc(_ohlc(a, [1.0, 1.0, 1.0]), _ohlc(b, [2.0, 2.0, 2.0]))
    assert len(mid) == 2  # two overlapping minutes


def test_derive_ohlc_is_ratio():
    idx = pd.date_range("2020-11-16", periods=2, freq="min", tz="UTC")
    num = _ohlc(idx, [10.0, 12.0])   # e.g. USDSEK
    den = _ohlc(idx, [8.0, 8.0])     # e.g. USDNOK
    cross = derive_ohlc(num, den)    # NOKSEK = USDSEK/USDNOK
    assert list(cross["close"]) == [1.25, 1.5]


def test_to_canonical_schema_and_vwap():
    idx = pd.date_range("2020-11-16", periods=2, freq="min", tz="UTC")
    out = to_canonical(_ohlc(idx, [1.10, 1.11]))
    assert list(out.columns) == CANONICAL_OHLCV_SCHEMA
    assert list(out["vwap"]) == [1.10, 1.11]      # vwap approximated as close
    assert list(out["trade_count"]) == [0, 0]
    assert out["timestamp"].dt.tz is not None


def test_to_canonical_empty_returns_empty():
    # Some instruments (restricted EM) return an empty, all-object frame for a
    # month Dukascopy lacks; to_canonical must yield an empty canonical frame,
    # not crash on volume.round().astype(int).
    out = to_canonical(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    assert list(out.columns) == CANONICAL_OHLCV_SCHEMA
    assert len(out) == 0


def test_to_canonical_coerces_object_volume():
    idx = pd.date_range("2020-11-16", periods=2, freq="min", tz="UTC")
    ohlc = _ohlc(idx, [1.10, 1.11])
    ohlc["volume"] = ohlc["volume"].astype(object)
    out = to_canonical(ohlc)
    assert list(out["volume"]) == [1, 1]


def test_derived_crosses_use_available_legs():
    # The 3 non-direct crosses must derive from instruments that ARE in the map.
    for x in ("NOKSEK", "NOKJPY", "SEKJPY"):
        spec = DUKA_MAP[x]
        assert isinstance(spec, tuple) and spec[0] == "derive"
