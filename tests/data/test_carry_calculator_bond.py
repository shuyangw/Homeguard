from datetime import date
import pytest
import src.data.carry_calculator as cc
from src.data.carry_calculator import CarryCalculator


def _patch_contracts(monkeypatch):
    # avoid needing real per-contract data: bond branch ignores front/second,
    # but compute() resolves them up top. Return a valid 3-month gap.
    monkeypatch.setattr(
        CarryCalculator, "_find_front_second_close",
        lambda self, root, d: (f"{root}H4", 100.0, f"{root}M4", 100.0))
    monkeypatch.setattr(CarryCalculator, "_months_between", lambda self, a, b, r: 3)


def test_zn_bond_carry_uses_cmt_minus_funding(monkeypatch):
    _patch_contracts(monkeypatch)
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: {"DGS10": 4.2, "DFF": 5.3}[sid])
    # ZN duration 9: 9*(4.2-5.3)/100 = -0.099
    got = CarryCalculator().compute("ZN", "bond", date(2024, 1, 3))
    assert abs(got - (9.0 * (4.2 - 5.3) / 100.0)) < 1e-9
    assert got < 0  # inverted curve -> negative carry (short)


def test_positive_curve_gives_positive_carry(monkeypatch):
    _patch_contracts(monkeypatch)
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: {"DGS30": 3.5, "DFF": 0.1}[sid])
    got = CarryCalculator().compute("ZB", "bond", date(2013, 6, 3))  # ZB->DGS30, dur 17
    assert got > 0 and abs(got - (17.0 * (3.5 - 0.1) / 100.0)) < 1e-9


def test_tenor_map_covers_all_price_traded(monkeypatch):
    _patch_contracts(monkeypatch)
    seen = {}
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: seen.setdefault(sid, 4.0) or 4.0)
    for root in ["ZT", "ZF", "ZN", "TN", "ZB", "UB"]:
        v = CarryCalculator().compute(root, "bond", date(2024, 1, 3))
        assert v is not None
    assert {"DGS2", "DGS5", "DGS10", "DGS30", "DFF"} <= set(seen)


def test_micro_yield_path_unchanged(monkeypatch):
    # 10Y is a MICRO_YIELD_ROOT -> must use derive_sofr, NOT get_fred_series.
    _patch_contracts(monkeypatch)
    def _boom(*a, **k):
        raise AssertionError("micro-yield path must not call get_fred_series")
    monkeypatch.setattr(cc, "get_fred_series", _boom)
    monkeypatch.setattr(cc, "derive_sofr", lambda d: 5.0)
    # front close (100.0 from _patch) treated as the yield for micro roots
    got = CarryCalculator().compute("10Y", "bond", date(2024, 1, 3))
    assert got == 9.0 * (100.0 - 5.0) / 100.0  # duration 9 for 10Y
