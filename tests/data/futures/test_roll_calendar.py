from datetime import date

import pytest

from src.data.futures.roll_calendar import (
    apply_fnd_clamp,
    detect_rolls,
    NoActiveContractError,
    RollCalendar,
    RollEvent,
)


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
