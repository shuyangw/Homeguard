"""Tests for the futures expectations module."""
from datetime import date

from src.data.validation.futures.expectations import (
    EXPECTED_DENSITY,
    EXPECTED_RANGES,
    EXPECTED_SCHEMAS,
    KNOWN_EVENTS,
    KNOWN_GAPS,
    LISTING_DATES,
)


def test_density_covers_all_53_continuous_symbols():
    expected_universe = {
        "ES", "NQ", "YM", "RTY", "MES", "MNQ", "M2K", "MYM",
        "CL", "NG", "HO", "RB", "BZ", "MCL", "MNG",
        "GC", "SI", "HG", "PL", "MGC", "SIL",
        "ZT", "ZF", "ZN", "TN", "ZB", "UB", "SR3", "SR1",
        "10Y", "30Y", "5YY", "2YY",
        "6E", "6J", "6B", "6A", "6C", "6S", "6N", "6M",
        "ZC", "ZS", "ZW", "KE", "ZL", "ZM", "LE", "HE",
        "BTC", "MBT", "ETH", "MET",
    }
    assert set(EXPECTED_DENSITY.keys()) == expected_universe


def test_density_ranges_well_formed():
    for sym, (lo, hi) in EXPECTED_DENSITY.items():
        assert lo > 0, f"{sym}: lower bound must be positive"
        assert lo < hi, f"{sym}: lo {lo} must be < hi {hi}"


def test_listing_dates_post_2010():
    for sym, d in LISTING_DATES.items():
        assert d >= date(2010, 6, 6), f"{sym}: listing {d} pre-dates dataset floor"


def test_known_gaps_well_ordered():
    for start, end, _reason in KNOWN_GAPS:
        assert start <= end


def test_expected_schemas_for_each_section():
    for section in [
        "futures_1min", "futures_per_contract_1min", "futures_options_1min",
        "futures_definitions", "futures_statistics", "futures_1min_oi_roll",
    ]:
        assert section in EXPECTED_SCHEMAS, f"missing schema for {section}"
        cols = EXPECTED_SCHEMAS[section]
        assert any(c[0] == "timestamp" for c in cols), \
            f"{section}: missing timestamp column"


def test_known_events_have_dates_and_descriptions():
    assert len(KNOWN_EVENTS) > 0
    for d, (sym, desc) in KNOWN_EVENTS.items():
        assert isinstance(d, date)
        assert isinstance(sym, str) and len(sym) > 0
        assert isinstance(desc, str) and len(desc) > 0
