from datetime import date

import polars as pl
import pytest

from src.data.futures.roll_calendar import (
    apply_fnd_clamp,
    detect_rolls,
    NoActiveContractError,
    RollCalendar,
    RollEvent,
)
from src.data.roll_detector import FuturesRollManager


def _day(n): return date(2024, 1, n)


def test_oi_crossover_with_hysteresis():
    # GCG4 front until Jan 10; GCJ4 OI overtakes for 2 consecutive days -> roll on 2nd
    oi = {
        _day(8):  {"GCG4": 100, "GCJ4": 10},
        _day(9):  {"GCG4": 90,  "GCJ4": 40},
        _day(10): {"GCG4": 50,  "GCJ4": 60},   # crossover day 1 (not yet, hysteresis=2)
        _day(11): {"GCG4": 30,  "GCJ4": 80},   # crossover day 2 -> ROLL here
        _day(12): {"GCG4": 20,  "GCJ4": 90},
    }
    rolls = detect_rolls("GC", oi, hysteresis=2)
    assert len(rolls) == 1
    assert rolls[0].roll_date == _day(11)
    assert rolls[0].from_symbol == "GCG4"
    assert rolls[0].to_symbol == "GCJ4"
    assert rolls[0].trigger == "oi_crossover"


def test_single_day_oi_blip_does_not_roll():
    # One-day OI spike in back month must NOT trigger a roll (hysteresis guards it)
    oi = {
        _day(8):  {"GCG4": 100, "GCJ4": 10},
        _day(9):  {"GCG4": 40,  "GCJ4": 60},   # blip up
        _day(10): {"GCG4": 90,  "GCJ4": 20},   # back to front dominant
        _day(11): {"GCG4": 85,  "GCJ4": 25},
    }
    rolls = detect_rolls("GC", oi, hysteresis=2)
    assert rolls == []


def test_fnd_clamp_pulls_physical_roll_earlier():
    # A physical root whose OI-roll lands AFTER the FND cutoff must be clamped earlier.
    rolls = [RollEvent(date(2024, 1, 28), "GCF4", "GCG4", "oi_crossover")]
    expirations = {"GCF4": date(2024, 1, 29)}   # last-trade; FND well before
    # GC fnd_offset_days=3 -> cutoff = expiration - 3 business days = ~2024-01-24
    clamped = apply_fnd_clamp("GC", rolls, expirations)
    assert clamped[0].roll_date <= date(2024, 1, 25)
    assert clamped[0].trigger == "fnd_clamp"


def test_fnd_clamp_noop_for_financial_root():
    rolls = [RollEvent(date(2024, 3, 15), "ESH4", "ESM4", "oi_crossover")]
    expirations = {"ESH4": date(2024, 3, 15)}
    clamped = apply_fnd_clamp("ES", rolls, expirations)
    assert clamped == rolls   # financial -> untouched


def test_missing_root_lookup_raises(tmp_path):
    cal = RollCalendar(cache_dir=tmp_path)   # empty cache
    with pytest.raises(NoActiveContractError):
        cal.get_front("GC", date(2024, 1, 15))


def test_roll_calendar_roundtrip(tmp_path):
    # Hand-write a 2-row calendar and confirm the lookup API reads it back.
    df = pl.DataFrame({
        "date": [date(2024, 1, 15), date(2024, 1, 16)],
        "front_symbol": ["GCG4", "GCG4"],
        "front_expiration": [date(2024, 2, 27), date(2024, 2, 27)],
        "front_activation": [date(2022, 3, 30), date(2022, 3, 30)],
        "next_cycle_symbol": ["GCH4", "GCH4"],
        "next_oi_symbol": ["GCJ4", "GCJ4"],
        "dte_front": [43, 42],
        "roll_trigger": ["oi_crossover", "oi_crossover"],
    })
    df.write_parquet(tmp_path / "GC.parquet")
    cal = RollCalendar(cache_dir=tmp_path)
    assert cal.get_front("GC", date(2024, 1, 15)).raw_symbol == "GCG4"
    assert cal.get_nth_by_cycle("GC", date(2024, 1, 15), 1).raw_symbol == "GCH4"
    assert cal.get_nth_by_oi("GC", date(2024, 1, 15), 1).raw_symbol == "GCJ4"
    assert cal.days_to_expiry("GC", date(2024, 1, 16)) == 42


def test_upcoming_rolls_from_calendar(tmp_path, monkeypatch):
    df = pl.DataFrame({
        "date": [date(2024, 1, 24), date(2024, 1, 25), date(2024, 1, 26)],
        "front_symbol": ["GCG4", "GCJ4", "GCJ4"],
        "front_expiration": [date(2024, 2, 27)] * 3,
        "front_activation": [date(2022, 3, 30)] * 3,
        "next_cycle_symbol": ["GCH4", "GCK4", "GCK4"],
        "next_oi_symbol": ["GCJ4", "GCM4", "GCM4"],
        "dte_front": [34, 33, 32],
        "roll_trigger": ["hold", "oi_crossover", "hold"],
    })
    df.write_parquet(tmp_path / "GC.parquet")
    monkeypatch.setattr(
        "src.data.roll_detector.roll_calendar_dir", lambda: tmp_path, raising=False,
    )
    mgr = FuturesRollManager(cache_dir=tmp_path)
    rolls = mgr.get_upcoming_rolls(["GC"], today=date(2024, 1, 20), lookahead_days=14)
    assert len(rolls) == 1
    assert rolls[0].to_contract == "GCJ4"
    assert rolls[0].roll_date == date(2024, 1, 25)
