"""Tests for the equity-options canonicalization layer.

Covers:
  1. Timestamp parsing (naive ET string -> Datetime[us, UTC]) incl. both DST edges
  2. `right` normalization PUT->P / CALL->C and loud failure on anything else
  3. `quote_valid` (V3) rule
  4. mid / spread_abs / spread_rel NULL exactly where quote_valid is False
  5. Snapshot guard: exact hit, fallback, and empty-session behaviour
  6. Snapshot guard never returns a bar after 15:45 ET (leakage guard)
  7. The `_eod` >= 1-session lag primitive (V6)
  8. dte / dte_trading incl. an expiry across a market holiday
  9. One real-data smoke test against SPY 2024-01 (skipped if H:\\ unavailable)
"""

from __future__ import annotations

import math
import time as _time
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.data.options import canonical as C

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _raw(rows):
    """Build a raw (on-disk schema) frame from dicts, filling defaults."""
    defaults = {
        "timestamp": "2024-01-02T15:45:00",
        "expiration": "2024-01-19",
        "strike": 470.0,
        "right": "CALL",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 10,
        "trade_count": 2,
        "vwap": 1.0,
        "bid_close": 1.0,
        "ask_close": 1.1,
        "implied_vol": 0.15,
        "delta": 0.5,
        "theta": -0.1,
        "vega": 0.2,
        "underlying_px": 470.5,
        "gamma_eod": 0.01,
        "open_interest_eod": 100,
    }
    full = [{**defaults, **r} for r in rows]
    schema = {
        "timestamp": pl.String,
        "expiration": pl.String,
        "strike": pl.Float64,
        "right": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int32,
        "trade_count": pl.Int32,
        "vwap": pl.Float64,
        "bid_close": pl.Float64,
        "ask_close": pl.Float64,
        "implied_vol": pl.Float64,
        "delta": pl.Float64,
        "theta": pl.Float64,
        "vega": pl.Float64,
        "underlying_px": pl.Float64,
        "gamma_eod": pl.Float64,
        "open_interest_eod": pl.Int64,
    }
    return pl.DataFrame(full, schema=schema)


# ------------------------------------------------- 0. on-disk schema variants


def test_expiration_accepts_date_dtype_variant():
    """100 partitions (SPY 2017-2018, QQQ 2017/2018/2023) ship `expiration` as
    date32 rather than string. Canonicalization must accept both."""
    raw = _raw([{"timestamp": "2024-01-02T15:45:00", "expiration": "2024-01-19"}])
    raw_date = raw.with_columns(pl.col("expiration").str.to_date())
    assert raw_date.schema["expiration"] == pl.Date

    out_str = C.canonicalize_frame(raw, root="SPY")
    out_date = C.canonicalize_frame(raw_date, root="SPY")

    assert out_date["expiry"][0] == date(2024, 1, 19)
    assert out_date.schema["expiry"] == pl.Date
    assert out_date["dte"][0] == out_str["dte"][0]
    assert out_date["dte_trading"][0] == out_str["dte_trading"][0]


# ---------------------------------------------------------------- 1. timestamps


def test_timestamp_parsed_to_utc_microsecond_datetime():
    df = C.canonicalize_frame(_raw([{"timestamp": "2024-01-02T15:45:00"}]), root="SPY")
    assert df.schema["ts"] == pl.Datetime("us", "UTC")
    assert df["ts"][0] == datetime(2024, 1, 2, 20, 45, tzinfo=UTC)  # EST = UTC-5


def test_timestamp_dst_spring_forward():
    # 2024-03-10 is the DST start; 2024-03-11 is EDT (UTC-4)
    df = C.canonicalize_frame(
        _raw(
            [
                {"timestamp": "2024-03-08T15:45:00"},  # EST, UTC-5
                {"timestamp": "2024-03-11T15:45:00"},  # EDT, UTC-4
            ]
        ),
        root="SPY",
    )
    assert df["ts"][0] == datetime(2024, 3, 8, 20, 45, tzinfo=UTC)
    assert df["ts"][1] == datetime(2024, 3, 11, 19, 45, tzinfo=UTC)


def test_timestamp_dst_fall_back():
    # 2024-11-03 is the DST end; 2024-11-04 is EST (UTC-5)
    df = C.canonicalize_frame(
        _raw(
            [
                {"timestamp": "2024-11-01T15:45:00"},  # EDT, UTC-4
                {"timestamp": "2024-11-04T15:45:00"},  # EST, UTC-5
            ]
        ),
        root="SPY",
    )
    assert df["ts"][0] == datetime(2024, 11, 1, 19, 45, tzinfo=UTC)
    assert df["ts"][1] == datetime(2024, 11, 4, 20, 45, tzinfo=UTC)


def test_session_date_is_eastern_local_date():
    df = C.canonicalize_frame(_raw([{"timestamp": "2024-07-05T15:45:00"}]), root="SPY")
    assert df.schema["session_date"] == pl.Date
    assert df["session_date"][0] == date(2024, 7, 5)


# ------------------------------------------------------------ 2. right mapping


def test_right_normalization():
    df = C.canonicalize_frame(
        _raw([{"right": "PUT"}, {"right": "CALL"}]), root="SPY"
    )
    assert df["right"].to_list() == ["P", "C"]


def test_unexpected_right_raises_loudly():
    with pytest.raises(C.UnexpectedRightError):
        C.canonicalize_frame(_raw([{"right": "X"}]), root="SPY")


# -------------------------------------------------------------- 3. quote_valid


def test_quote_valid_rule_v3():
    df = C.canonicalize_frame(
        _raw(
            [
                {"bid_close": 1.0, "ask_close": 1.1},  # good
                {"bid_close": 1.0, "ask_close": 1.0},  # locked -> ask >= bid ok
                {"bid_close": 1.2, "ask_close": 1.1},  # crossed -> False
                {"bid_close": 0.0, "ask_close": 0.1},  # zero bid -> False
                {"bid_close": -1.0, "ask_close": 0.1},  # negative bid -> False
                {"bid_close": float("nan"), "ask_close": 1.1},  # NaN -> False
                {"bid_close": 1.0, "ask_close": float("inf")},  # inf -> False
                {"bid_close": None, "ask_close": 1.1},  # null -> False
                {"bid_close": 1.0, "ask_close": None},  # null -> False
            ]
        ),
        root="SPY",
    )
    assert df["quote_valid"].to_list() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


# ------------------------------------------------ 4. derived cols null when invalid


def test_mid_and_spreads_null_exactly_where_invalid():
    df = C.canonicalize_frame(
        _raw(
            [
                {"bid_close": 1.0, "ask_close": 1.2},
                {"bid_close": 0.0, "ask_close": 0.2},
                {"bid_close": 1.3, "ask_close": 1.2},
            ]
        ),
        root="SPY",
    )
    valid = df["quote_valid"].to_list()
    for col in ("mid", "spread_abs", "spread_rel"):
        nulls = df[col].is_null().to_list()
        assert nulls == [not v for v in valid], col
    assert df["mid"][0] == pytest.approx(1.1)
    assert df["spread_abs"][0] == pytest.approx(0.2)
    assert df["spread_rel"][0] == pytest.approx(0.2 / 1.1)


def test_no_imputation_of_invalid_quotes():
    df = C.canonicalize_frame(_raw([{"bid_close": 0.0, "ask_close": 0.5}]), root="SPY")
    assert df["mid"][0] is None
    # raw bid/ask are preserved, not repaired
    assert df["bid"][0] == 0.0
    assert df["ask"][0] == 0.5


# ------------------------------------------------------------ 5/6. snapshot guard


def _day_bars(times, bids=None, asks=None):
    bids = bids or [1.0] * len(times)
    asks = asks or [1.1] * len(times)
    rows = [
        {"timestamp": f"2024-01-02T{t}", "bid_close": b, "ask_close": a}
        for t, b, a in zip(times, bids, asks)
    ]
    return C.canonicalize_frame(_raw(rows), root="SPY")


def test_snapshot_exact_1545_hit():
    bars = _day_bars(["15:43:00", "15:44:00", "15:45:00", "15:46:00"])
    snap = C.snapshot_from_bars(bars)
    assert snap.height == 1
    assert snap["snapshot_fallback"][0] is False
    assert snap["snapshot_ts"][0] == datetime(2024, 1, 2, 20, 45, tzinfo=UTC)


def test_snapshot_falls_back_to_latest_valid_bar():
    bars = _day_bars(
        ["15:40:00", "15:43:00", "15:45:00"],
        bids=[1.0, 1.0, 0.0],  # 15:45 has a zero bid -> invalid
        asks=[1.1, 1.1, 0.5],
    )
    snap = C.snapshot_from_bars(bars)
    assert snap.height == 1
    assert snap["snapshot_fallback"][0] is True
    assert snap["snapshot_ts"][0] == datetime(2024, 1, 2, 20, 43, tzinfo=UTC)
    # snapshot_ts carries the ACTUAL bar time, never backfilled to 15:45
    assert snap["ts"][0] == snap["snapshot_ts"][0]


def test_snapshot_emits_no_record_when_no_valid_bar():
    bars = _day_bars(["15:40:00", "15:45:00"], bids=[0.0, 0.0], asks=[0.5, 0.5])
    snap = C.snapshot_from_bars(bars)
    assert snap.height == 0


def test_snapshot_never_returns_a_bar_after_1545():
    # only bars AFTER the snapshot minute exist -> nothing may be emitted
    bars = _day_bars(["15:46:00", "15:50:00", "16:00:00"])
    assert C.snapshot_from_bars(bars).height == 0

    # a later, valid bar must never win over an earlier one
    bars = _day_bars(["15:30:00", "15:59:00", "16:00:00"])
    snap = C.snapshot_from_bars(bars)
    assert snap.height == 1
    assert snap["snapshot_ts"][0] == datetime(2024, 1, 2, 20, 30, tzinfo=UTC)


def test_snapshot_time_is_1545_et():
    assert C.SNAPSHOT_TIME_ET == time(15, 45)


def test_snapshot_one_row_per_contract_session():
    rows = []
    for t in ("15:44:00", "15:45:00"):
        for strike, right in ((470.0, "CALL"), (470.0, "PUT"), (480.0, "CALL")):
            rows.append(
                {
                    "timestamp": f"2024-01-02T{t}",
                    "strike": strike,
                    "right": right,
                }
            )
        for strike, right in ((470.0, "CALL"),):
            rows.append(
                {
                    "timestamp": f"2024-01-03T{t}",
                    "strike": strike,
                    "right": right,
                }
            )
    snap = C.snapshot_from_bars(C.canonicalize_frame(_raw(rows), root="SPY"))
    assert snap.height == 4
    key = ["root", "expiry", "strike", "right", "session_date"]
    assert snap.select(key).unique().height == 4


def test_snapshot_respects_early_close_cutoff():
    # On a 13:00 early close the vendor pads bars to 16:00 with stale quotes.
    bars = _day_bars(["12:59:00", "13:00:00", "15:45:00"])
    snap = C.snapshot_from_bars(bars, session_close_et=time(13, 0))
    assert snap.height == 1
    assert snap["snapshot_ts"][0] == datetime(2024, 1, 2, 18, 0, tzinfo=UTC)
    assert snap["snapshot_fallback"][0] is True


# ------------------------------------------------------------- 7. _eod lag (V6)


def _eod_series(sessions_and_oi):
    rows = [
        {
            "timestamp": f"{d}T15:45:00",
            "open_interest_eod": oi,
            "gamma_eod": float(oi) / 1000.0,
        }
        for d, oi in sessions_and_oi
    ]
    return C.canonicalize_frame(_raw(rows), root="SPY")


def test_eod_lag_equals_prior_session_value():
    df = C.add_eod_lag(
        _eod_series(
            [("2024-01-02", 100), ("2024-01-03", 200), ("2024-01-04", 300)]
        )
    )
    df = df.sort("session_date")
    assert df["oi_eod_lag1"].to_list() == [None, 100, 200]
    assert df["gamma_eod_lag1"].to_list() == [None, 0.1, 0.2]
    # raw same-session (leak-bearing) values remain present
    assert df["oi_eod"].to_list() == [100, 200, 300]


def test_eod_lag_first_session_is_null():
    df = C.add_eod_lag(_eod_series([("2024-01-02", 100)]))
    assert df["oi_eod_lag1"][0] is None
    assert df["gamma_eod_lag1"][0] is None


def test_eod_lag_does_not_carry_across_a_session_gap():
    # 2024-01-03 is missing for this contract -> 01-04's lag must be NULL,
    # NOT 01-02's stale value (registered: no forward-fill across a gap).
    df = C.add_eod_lag(
        _eod_series([("2024-01-02", 100), ("2024-01-04", 300), ("2024-01-05", 400)])
    ).sort("session_date")
    assert df["oi_eod_lag1"].to_list() == [None, None, 300]


def test_eod_lag_is_per_contract():
    rows = [
        {"timestamp": "2024-01-02T15:45:00", "strike": 470.0, "open_interest_eod": 100},
        {"timestamp": "2024-01-03T15:45:00", "strike": 470.0, "open_interest_eod": 110},
        {"timestamp": "2024-01-02T15:45:00", "strike": 480.0, "open_interest_eod": 500},
        {"timestamp": "2024-01-03T15:45:00", "strike": 480.0, "open_interest_eod": 550},
    ]
    df = C.add_eod_lag(C.canonicalize_frame(_raw(rows), root="SPY"))
    got = {
        (r["strike"], r["session_date"]): r["oi_eod_lag1"] for r in df.to_dicts()
    }
    assert got[(470.0, date(2024, 1, 3))] == 100
    assert got[(480.0, date(2024, 1, 3))] == 500
    assert got[(470.0, date(2024, 1, 2))] is None


def test_eod_lag_weekend_is_adjacent():
    # Fri 2024-01-05 -> Mon 2024-01-08 are ADJACENT trading sessions
    df = C.add_eod_lag(
        _eod_series([("2024-01-05", 100), ("2024-01-08", 200)])
    ).sort("session_date")
    assert df["oi_eod_lag1"].to_list() == [None, 100]


def test_canonical_schema_preserves_eod_suffix():
    df = C.canonicalize_frame(_raw([{}]), root="SPY")
    cols = set(df.columns)
    assert "oi_eod" in cols
    assert "gamma_eod" in cols
    # the data_loader.py footgun (stripping the marker) must never reappear
    assert "open_interest" not in cols
    assert "gamma" not in cols
    assert "oi_eod" in C.CANONICAL_COLUMNS
    assert "gamma_eod" in C.CANONICAL_COLUMNS
    assert "open_interest" not in C.CANONICAL_COLUMNS
    assert "gamma" not in C.CANONICAL_COLUMNS


# ---------------------------------------------------------------- 8. dte columns


def test_dte_calendar_and_trading():
    df = C.canonicalize_frame(
        _raw(
            [
                {"timestamp": "2024-01-02T15:45:00", "expiration": "2024-01-02"},
                {"timestamp": "2024-01-02T15:45:00", "expiration": "2024-01-03"},
                {"timestamp": "2024-01-05T15:45:00", "expiration": "2024-01-08"},
            ]
        ),
        root="SPY",
    )
    assert df["dte"].to_list() == [0, 1, 3]
    # Fri -> Mon is one trading session
    assert df["dte_trading"].to_list() == [0, 1, 1]


def test_dte_trading_skips_market_holiday():
    # Memorial Day 2024-05-27 (Mon) is a holiday.
    # Thu 2024-05-23 -> Tue 2024-05-28 : calendar 5, trading sessions {24, 28} = 2
    df = C.canonicalize_frame(
        _raw([{"timestamp": "2024-05-23T15:45:00", "expiration": "2024-05-28"}]),
        root="SPY",
    )
    assert df["dte"][0] == 5
    assert df["dte_trading"][0] == 2


def test_dte_trading_skips_good_friday():
    # Good Friday 2024-03-29. Thu 2024-03-28 -> Mon 2024-04-01: calendar 4, trading 1
    df = C.canonicalize_frame(
        _raw([{"timestamp": "2024-03-28T15:45:00", "expiration": "2024-04-01"}]),
        root="SPY",
    )
    assert df["dte"][0] == 4
    assert df["dte_trading"][0] == 1


# ------------------------------------------------------------- 9. real-data smoke


def _spy_jan_2024_path():
    try:
        from src.settings import get_local_storage_dir

        return (
            get_local_storage_dir()
            / "options"
            / "options_combined"
            / "root=SPY"
            / "year=2024"
            / "month=01"
            / "data.parquet"
        )
    except Exception:
        return None


_SPY = _spy_jan_2024_path()
_HAVE_REAL = _SPY is not None and _SPY.exists()


@pytest.mark.skipif(not _HAVE_REAL, reason="options store (H:) unavailable")
def test_real_data_smoke_spy_2024_01():
    batches = C.iter_canonical_batches("SPY", 2024, 1, max_batches=1)
    df = next(iter(batches))
    assert set(C.CANONICAL_COLUMNS).issubset(set(df.columns))
    assert df.schema["ts"] == pl.Datetime("us", "UTC")
    assert set(df["right"].unique().to_list()).issubset({"C", "P"})
    assert df["root"][0] == "SPY"
    # marks are never trade-derived
    assert df["mid"].is_null().to_list() == [not v for v in df["quote_valid"].to_list()]


@pytest.mark.skipif(not _HAVE_REAL, reason="options store (H:) unavailable")
def test_real_data_snapshot_never_after_1545_spy_2024_01():
    snap = C.build_chain_eod_frame("SPY", 2024, 1)
    assert snap.height > 0
    local = snap["snapshot_ts"].dt.convert_time_zone("America/New_York")
    assert local.dt.time().max() <= C.SNAPSHOT_TIME_ET
    key = ["root", "expiry", "strike", "right", "session_date"]
    assert snap.select(key).unique().height == snap.height
    assert snap["quote_valid"].all()
