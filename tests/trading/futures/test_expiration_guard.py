"""Tests for ExpirationGuard."""
from datetime import date, datetime, timezone

import pytest

from src.trading.futures.expiration_guard import (
    EXPIRATION_THRESHOLDS,
    ExpirationGuard,
    ExpirationVerdict,
)
from src.trading.futures.position import FuturesPosition


def _today_is(d: date, monkeypatch):
    """Helper to freeze today() to a known date inside ExpirationGuard."""
    monkeypatch.setattr(
        "src.trading.futures.expiration_guard._today",
        lambda: d,
    )


def test_thresholds_cover_all_53_continuous_universe():
    """Every symbol the strategies trade should have a threshold (or fall back to _DEFAULT)."""
    expected_roots = {
        "ES", "NQ", "YM", "RTY", "MES", "MNQ", "M2K", "MYM",
        "CL", "NG", "HO", "RB", "BZ", "MCL", "MNG",
        "GC", "SI", "HG", "PL", "MGC", "SIL",
        "ZT", "ZF", "ZN", "TN", "ZB", "UB", "SR3", "SR1",
        "10Y", "30Y", "5YY", "2YY",
        "6E", "6J", "6B", "6A", "6C", "6S", "6N", "6M",
        "ZC", "ZS", "ZW", "KE", "ZL", "ZM", "LE", "HE",
        "BTC", "MBT", "ETH", "MET",
    }
    for root in expected_roots:
        # Either has explicit threshold or falls to _DEFAULT
        threshold = EXPIRATION_THRESHOLDS.get(root, EXPIRATION_THRESHOLDS["_DEFAULT"])
        assert threshold > 0
    assert "_DEFAULT" in EXPIRATION_THRESHOLDS


def test_check_new_entry_far_from_expiry_returns_ok(monkeypatch):
    """Expiration 30 days out, threshold 5 -> OK."""
    _today_is(date(2026, 5, 1), monkeypatch)
    guard = ExpirationGuard()
    v = guard.check_new_entry_with_expiration("ES", date(2026, 6, 1))
    assert v == ExpirationVerdict.OK


def test_check_new_entry_within_threshold_returns_warn(monkeypatch):
    """ES threshold is 5 days. Expiration 3 days out -> WARN (block new entries)."""
    _today_is(date(2026, 5, 28), monkeypatch)
    guard = ExpirationGuard()
    v = guard.check_new_entry_with_expiration("ES", date(2026, 6, 1))
    assert v == ExpirationVerdict.WARN


def test_check_new_entry_expired_returns_expired(monkeypatch):
    """Expiration in the past -> EXPIRED."""
    _today_is(date(2026, 6, 5), monkeypatch)
    guard = ExpirationGuard()
    v = guard.check_new_entry_with_expiration("ES", date(2026, 6, 1))
    assert v == ExpirationVerdict.EXPIRED


def test_check_existing_position_must_roll_or_close_at_1_day(monkeypatch):
    """Existing position with 1 day to expiry must take action."""
    _today_is(date(2026, 5, 31), monkeypatch)
    pos = FuturesPosition(
        symbol_root="ES", contract_month="202606", raw_symbol="ESM6",
        quantity=2, avg_entry_price=5300.0,
        multiplier=5.0, tick_size=0.25, tick_value=1.25,
        expiration_date=date(2026, 6, 1),
        broker="ibkr_futures", strategy="x",
        opened_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    guard = ExpirationGuard()
    v = guard.check_existing_position(pos)
    assert v == ExpirationVerdict.MUST_ROLL_OR_CLOSE


def test_check_existing_position_within_threshold_warn(monkeypatch):
    """Existing position with 3 days to ES expiry returns WARN."""
    _today_is(date(2026, 5, 29), monkeypatch)
    pos = FuturesPosition(
        symbol_root="ES", contract_month="202606", raw_symbol="ESM6",
        quantity=2, avg_entry_price=5300.0,
        multiplier=5.0, tick_size=0.25, tick_value=1.25,
        expiration_date=date(2026, 6, 1),
        broker="ibkr_futures", strategy="x",
        opened_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    guard = ExpirationGuard()
    v = guard.check_existing_position(pos)
    assert v == ExpirationVerdict.WARN


def test_check_existing_position_far_ok(monkeypatch):
    """30 days to expiry -> OK for existing position."""
    _today_is(date(2026, 5, 1), monkeypatch)
    pos = FuturesPosition(
        symbol_root="ES", contract_month="202606", raw_symbol="ESM6",
        quantity=2, avg_entry_price=5300.0,
        multiplier=5.0, tick_size=0.25, tick_value=1.25,
        expiration_date=date(2026, 6, 1),
        broker="ibkr_futures", strategy="x",
        opened_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    assert ExpirationGuard().check_existing_position(pos) == ExpirationVerdict.OK


def test_per_family_threshold_varies():
    """Rates (ZT) has shorter threshold than equity index (ES)."""
    assert EXPIRATION_THRESHOLDS["ES"] > EXPIRATION_THRESHOLDS["ZT"]
    assert EXPIRATION_THRESHOLDS["LE"] >= EXPIRATION_THRESHOLDS["GC"]  # livestock: physical delivery
