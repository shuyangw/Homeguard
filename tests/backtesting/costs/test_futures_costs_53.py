"""Coverage test: futures cost model must price all 53 SPECS roots."""
from __future__ import annotations

from src.data.futures.contract_specs import SPECS
from src.backtesting.costs.futures import futures_round_trip_usd, PER_SIDE_COMMISSION_USD


def test_all_53_roots_priced():
    for root in SPECS:
        rt = futures_round_trip_usd(root, regular_hours=True, n_contracts=1)
        assert rt > 0, f"{root} round-trip cost not positive"


def test_commission_covers_all_roots():
    assert set(PER_SIDE_COMMISSION_USD) >= set(SPECS)


def test_micro_cheaper_than_full():
    # MES round-trip should be well below ES
    assert futures_round_trip_usd("MES") < futures_round_trip_usd("ES")
