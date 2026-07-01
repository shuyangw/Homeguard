from datetime import date

from scripts.data.build_roll_calendar import _cycle_order_key, _next_by_cycle


def test_cycle_order_key_single_and_two_digit_year_agree():
    assert _cycle_order_key("GCG4", "GC") == _cycle_order_key("GCG24", "GC")


def test_cycle_order_key_orders_within_year():
    assert _cycle_order_key("GCG4", "GC") < _cycle_order_key("GCJ4", "GC")


def test_next_by_cycle_skips_off_cycle_contract():
    # GC liquid cycle is GJMQVZ (Feb,Apr,Jun,Aug,Oct,Dec). GCF4 (Jan) is off-cycle
    # and must be skipped; next after GCG4 (Feb) is GCJ4 (Apr), not GCF4.
    day_oi = {"GCG4": 100, "GCF4": 90, "GCJ4": 50}
    assert _next_by_cycle(day_oi, "GCG4", "GC", next_oi_fallback="GCJ4") == "GCJ4"


def test_next_by_cycle_falls_back_when_front_off_cycle():
    day_oi = {"GCF4": 100, "GCJ4": 50}  # front GCF4 is off-cycle
    assert _next_by_cycle(day_oi, "GCF4", "GC", next_oi_fallback="GCJ4") == "GCJ4"
