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
    gross = MarginModel(offset_matrix={}).requirement({"ES": 1, "NQ": -1})
    netted = MarginModel().requirement({"ES": 1, "NQ": -1})   # default ES/NQ offset applies
    assert netted < gross  # opposite-signed offset pair gets a credit


def test_offset_not_applied_same_direction():
    same_default = MarginModel().requirement({"ES": 1, "NQ": 1})
    no_offset = MarginModel(offset_matrix={}).requirement({"ES": 1, "NQ": 1})
    assert same_default == pytest.approx(no_offset)  # same-direction -> no offset


def test_check_and_scale_pro_rata():
    m = MarginModel()
    # targets requiring far more than the cap -> scaled down
    targets = {"ES": 10}
    scaled = m.check_and_scale(targets, equity=10_000, cap=0.5)
    assert 0 <= scaled["ES"] < 10
    assert m.requirement(scaled) <= 0.5 * 10_000 + get_spec_init("ES")  # within one contract of cap


def test_check_and_scale_keeps_positive_and_within_budget():
    from src.data.futures.contract_specs import get_spec

    m = MarginModel()
    # 10 MES -> req 16000; equity 10000, cap 0.5 -> budget 5000; factor 0.3125 -> 3 contracts
    scaled = m.check_and_scale({"MES": 10}, equity=10_000, cap=0.5)
    assert scaled["MES"] == 3
    req = m.requirement(scaled)
    assert 0 < req <= 5000
    assert req > 5000 - get_spec("MES").initial_margin   # within one contract of the budget


def test_utilization_ratio():
    m = MarginModel()
    req = m.requirement({"MES": 2})
    assert m.utilization({"MES": 2}, equity=req * 2) == pytest.approx(0.5)
