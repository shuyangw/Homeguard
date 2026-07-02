import pytest

from src.backtesting.margin.futures_margin import MarginModel
from src.data.futures.contract_specs import get_spec


def get_spec_init(root: str) -> float:
    return get_spec(root).initial_margin


def test_requirement_sums_scan_range():
    m = MarginModel()
    # 2 MES (init ~1600 each) -> ~3200; exact value read from specs
    exp = 2 * get_spec("MES").initial_margin
    assert m.requirement({"MES": 2}) == pytest.approx(exp)


def test_offset_credit_reduces_requirement():
    m = MarginModel(offset_matrix={("ES", "NQ"): 0.75})
    gross = m.__class__().requirement({"ES": 1, "NQ": -1})
    netted = m.requirement({"ES": 1, "NQ": -1})
    assert netted < gross  # opposite-signed offset pair gets a credit


def test_offset_not_applied_same_direction():
    m = MarginModel(offset_matrix={("ES", "NQ"): 0.75})
    same = m.requirement({"ES": 1, "NQ": 1})
    none_m = MarginModel().requirement({"ES": 1, "NQ": 1})
    assert same == pytest.approx(none_m)  # same-direction -> no offset


def test_check_and_scale_pro_rata():
    m = MarginModel()
    # targets requiring far more than the cap -> scaled down
    targets = {"ES": 10}
    scaled = m.check_and_scale(targets, equity=10_000, cap=0.5)
    assert 0 <= scaled["ES"] < 10
    assert m.requirement(scaled) <= 0.5 * 10_000 + get_spec_init("ES")  # within one contract of cap
