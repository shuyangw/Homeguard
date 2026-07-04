from datetime import date
import pytest
from src.data.futures.asset_class import asset_class_for, cluster_for


def test_crypto_maps():
    for r in ("BTC", "ETH"):
        assert asset_class_for(r) == "crypto"
        assert cluster_for(r) == "crypto"


def test_crypto_carry_is_annualized_roll_yield(monkeypatch):
    from src.data import carry_calculator as cc
    calc = cc.CarryCalculator()
    monkeypatch.setattr(calc, "_find_front_second_close",
                        lambda root, d: ("BTCF4", 100.0, "BTCG4", 102.0))
    val = calc.compute("BTC", "crypto", date(2024, 1, 15))
    assert val == pytest.approx((102.0 - 100.0) / 100.0 * (365.0 / 30.0))
