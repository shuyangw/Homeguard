"""Tests for M2 -- the mark convention (spec v2 Section 4 M2, amended by A3)."""
import numpy as np
import pandas as pd
import pytest

from src.backtesting.options.marks import (
    A3_RAW_MARK_CANDIDATES,
    LOW_DELTA_XCHECK_THRESHOLD,
    PROHIBITED_MARK_COLUMNS,
    SURFACE_DIVERGENCE_VOL_PTS,
    MarkSourceError,
    ProhibitedMarkError,
    REJECT_CROSSED,
    REJECT_NONFINITE,
    REJECT_ZERO_BID,
    REJECT_ZERO_QUOTE_OPEN_BAR,
    assert_mark_source_allowed,
    assert_not_trade_print,
    greek_usable_mask,
    mark_chain,
)


def _chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root": ["SPY"] * 5,
            "session_date": [pd.Timestamp("2024-06-03").date()] * 5,
            "expiry": [pd.Timestamp("2024-07-19").date()] * 5,
            "strike": [500.0, 510.0, 520.0, 530.0, 540.0],
            "right": ["C"] * 5,
            "bid": [10.00, 0.00, 5.00, np.nan, 1.00],
            "ask": [10.20, 0.05, 4.90, 2.00, 1.10],
            "delta": [0.60, 0.30, 0.20, 0.10, 0.04],
            "implied_vol": [0.15, 0.16, 0.17, 0.18, 0.30],
            "iv_smooth": [0.15, 0.16, 0.17, 0.18, 0.20],
            "quote_valid": [True, False, False, False, True],
        }
    )


def test_valid_quote_marks_at_mid():
    out = mark_chain(_chain())
    assert out.loc[0, "mark"] == pytest.approx(10.10)
    assert bool(out.loc[0, "mark_valid"])


def test_zero_bid_is_flagged_not_dropped():
    src = _chain()
    out = mark_chain(src)
    assert len(out) == len(src), "invalid rows must be FLAGGED, never dropped"
    assert out.loc[1, "mark_reject_reason"] == REJECT_ZERO_BID
    assert not bool(out.loc[1, "mark_valid"])
    assert not np.isfinite(out.loc[1, "mark"])


def test_crossed_quote_is_rejected():
    out = mark_chain(_chain())
    assert out.loc[2, "mark_reject_reason"] == REJECT_CROSSED


def test_nonfinite_quote_is_rejected():
    out = mark_chain(_chain())
    assert out.loc[3, "mark_reject_reason"] == REJECT_NONFINITE


def test_no_imputation_anywhere():
    """A rejected row must stay NaN -- no forward-fill, no interpolation."""
    out = mark_chain(_chain())
    bad = ~out["mark_valid"].to_numpy(dtype=bool)
    assert np.all(~np.isfinite(out.loc[bad, "mark"].to_numpy(dtype=float)))


def test_the_0930_zero_quote_bar_gets_its_own_reason():
    """The 09:30 bar has bid == ask == 0 universally. It is a known structural
    artifact and must be named, not lumped in with a genuine zero bid."""
    df = _chain().iloc[:1].copy()
    df["bid"] = 0.0
    df["ask"] = 0.0
    df["quote_valid"] = False
    df["bar_time_et"] = [pd.Timestamp("2024-06-03 09:30:00").time()]
    out = mark_chain(df)
    assert out.loc[0, "mark_reject_reason"] == REJECT_ZERO_QUOTE_OPEN_BAR


def test_trade_print_columns_are_prohibited_as_marks():
    for col in ("close", "vwap", "open", "high", "low"):
        assert col in PROHIBITED_MARK_COLUMNS
        with pytest.raises(ProhibitedMarkError):
            assert_not_trade_print(col)
    assert_not_trade_print("mid")  # must not raise


def test_surface_divergence_is_flagged_but_the_mark_is_unchanged():
    """M2: 'a mid that diverges materially from the surface is FLAGGED, not
    silently used' -- and never replaced by the surface."""
    df = _chain()
    out = mark_chain(df)
    row = out.iloc[4]
    assert abs(row["delta"]) < LOW_DELTA_XCHECK_THRESHOLD
    # |0.30 - 0.20| = 10 vol points, well over the registered threshold
    assert row["surface_divergence_vol_pts"] == pytest.approx(10.0)
    assert bool(row["surface_divergence_flag"])
    assert row["mark"] == pytest.approx(1.05), "mark must remain the quote mid"


def test_surface_divergence_threshold_is_the_registered_one():
    assert SURFACE_DIVERGENCE_VOL_PTS == 1.5


def test_a3_forbids_iv_smooth_as_a_mark_source_for_047_and_030():
    assert A3_RAW_MARK_CANDIDATES == {"OPT-047", "OPT-030"}
    for cid in A3_RAW_MARK_CANDIDATES:
        with pytest.raises(MarkSourceError):
            assert_mark_source_allowed(cid, "iv_smooth")
        assert_mark_source_allowed(cid, "quote_mid")  # must not raise


def test_a3_does_not_relax_p1_for_other_candidates():
    """A3 Section 4: no slate-wide relaxation. OPT-027 sits at 0.10 delta
    exactly, so P1's < 0.10 rule does not bind and it is unaffected."""
    assert_mark_source_allowed("OPT-027", "quote_mid")
    assert_mark_source_allowed("OPT-015", "quote_mid")


def test_nan_greeks_are_detected_by_value_not_by_null_count():
    """The NaN-vs-NULL trap: a 100%-NaN float64 column reports null_count == 0."""
    df = _chain()
    df["delta"] = np.nan
    assert df["delta"].isnull().sum() == len(df)  # pandas sees NaN as null...
    arrow_null_count = pd.array(df["delta"].to_numpy(dtype=float)).size
    assert arrow_null_count == len(df)
    mask = greek_usable_mask(df, require_delta=True)
    assert not mask.any(), "an all-NaN delta column must be unusable"


def test_greek_usable_mask_rejects_implausible_iv():
    df = _chain()
    df["implied_vol"] = [0.15, 0.16, 0.17, 0.18, 9.99]
    mask = greek_usable_mask(df, require_delta=False)
    assert not bool(mask.iloc[4])
