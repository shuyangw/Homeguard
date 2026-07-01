import pytest
from src.data.futures.contract_specs import get_spec, SPECS


def test_gc_spec_physical():
    s = get_spec("GC")
    assert s.multiplier == 100.0        # $100/point full GC
    assert s.tick_size == 0.1
    assert s.settlement_type == "physical"
    assert s.fnd_offset_days > 0        # metals roll before FND


def test_es_spec_financial_no_fnd():
    s = get_spec("ES")
    assert s.multiplier == 50.0
    assert s.settlement_type == "financial"
    assert s.fnd_offset_days == 0       # cash-settled, no FND clamp


def test_all_53_roots_present():
    expected = {
        "ES","NQ","YM","RTY","MES","MNQ","M2K","MYM",
        "CL","NG","HO","RB","BZ","MCL","MNG",
        "GC","SI","HG","PL","MGC","SIL",
        "ZT","ZF","ZN","TN","ZB","UB","SR3","SR1","10Y","30Y","5YY","2YY",
        "6E","6J","6B","6A","6C","6S","6N","6M",
        "ZC","ZS","ZW","KE","ZL","ZM","LE","HE",
        "BTC","MBT","ETH","MET",
    }
    assert expected <= set(SPECS.keys())


def test_unknown_root_raises():
    with pytest.raises(KeyError):
        get_spec("XYZ")


def test_tick_value_arithmetic_consistency():
    # tick_value must equal multiplier * tick_size for every root (guards data-entry errors)
    for root, s in SPECS.items():
        assert s.tick_value == pytest.approx(s.multiplier * s.tick_size, abs=1e-6), (
            f"{root}: multiplier {s.multiplier} * tick_size {s.tick_size} "
            f"= {s.multiplier * s.tick_size}, but tick_value = {s.tick_value}"
        )


def test_margin_fields_present_and_ordered():
    from src.data.futures.contract_specs import SPECS
    for root, s in SPECS.items():
        assert s.initial_margin > 0, f"{root} initial_margin not positive"
        assert 0 < s.maintenance_margin <= s.initial_margin, f"{root} maintenance>{root} initial"


def test_micro_margin_below_full():
    from src.data.futures.contract_specs import get_spec
    assert get_spec("MES").initial_margin < get_spec("ES").initial_margin
