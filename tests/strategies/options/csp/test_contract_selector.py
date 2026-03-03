"""Tests for CSPContractSelector - options chain filtering for cash-secured puts."""

from datetime import date
from typing import Dict, List

import pandas as pd
import pytest

from src.strategies.options.csp.contract_selector import CSPContractSelector


DEFAULTS: Dict = {
    "option_type": "P",
    "days_to_expiry": 28,
    "delta": -0.30,
    "open_interest": 500,
    "bid": 2.50,
    "ask": 2.70,
    "mid_price": 2.60,
    "strike": 150.0,
    "underlying_price": 160.0,
    "implied_vol": 0.30,
    "gamma": 0.02,
    "theta": -0.05,
    "vega": 0.15,
    "expiry": date(2024, 7, 15),
    "volume": 200,
}


def _make_chain(rows: List[Dict]) -> pd.DataFrame:
    full_rows = []
    for row in rows:
        merged = {**DEFAULTS, **row}
        full_rows.append(merged)
    return pd.DataFrame(full_rows)


class TestCSPContractSelector:

    def test_selects_best_put(self):
        chain = _make_chain([
            {"strike": 150.0, "bid": 2.50, "ask": 2.70, "mid_price": 2.60},
            {"strike": 145.0, "bid": 1.80, "ask": 2.00, "mid_price": 1.90},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is not None
        assert result["strike"] == 150.0
        assert result["mid_price"] == 2.60

    def test_filters_calls(self):
        chain = _make_chain([
            {"option_type": "C", "delta": 0.30},
            {"option_type": "C", "delta": 0.45},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_filters_delta_out_of_range(self):
        chain = _make_chain([
            {"delta": -0.10},
            {"delta": -0.50},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_filters_dte_out_of_range(self):
        chain = _make_chain([
            {"days_to_expiry": 10},
            {"days_to_expiry": 50},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_filters_low_open_interest(self):
        chain = _make_chain([
            {"open_interest": 50},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_filters_wide_spread(self):
        chain = _make_chain([
            {"bid": 1.00, "ask": 2.00, "mid_price": 1.50},
        ])
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_empty_chain_returns_none(self):
        chain = pd.DataFrame()
        selector = CSPContractSelector()
        result = selector.select_contract(chain)

        assert result is None

    def test_custom_parameters(self):
        chain = _make_chain([
            {"delta": -0.40, "days_to_expiry": 45},
        ])
        selector = CSPContractSelector(
            target_delta_min=-0.45,
            target_delta_max=-0.35,
            min_dte=40,
            max_dte=50,
        )
        result = selector.select_contract(chain)

        assert result is not None
        assert result["delta"] == -0.40
