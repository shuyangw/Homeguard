"""Tests for CSPMarkToMarket - position pricing from chain data with B-S fallback."""

from datetime import date
from typing import Dict, List

import pandas as pd
import pytest

from src.strategies.options.csp.mark_to_market import CSPMarkToMarket
from src.strategies.options.csp.position import CSPPosition


CHAIN_DEFAULTS: Dict = {
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
    "expiry": date(2026, 7, 18),
    "volume": 200,
}


def _make_chain(rows: List[Dict]) -> pd.DataFrame:
    full_rows = []
    for row in rows:
        merged = {**CHAIN_DEFAULTS, **row}
        full_rows.append(merged)
    return pd.DataFrame(full_rows)


def _make_position(**overrides) -> CSPPosition:
    defaults = dict(
        symbol="AAPL",
        strike=150.0,
        expiry=date(2026, 7, 18),
        entry_date=date(2026, 6, 20),
        entry_price=2.50,
        num_contracts=1,
        collateral=15000.0,
    )
    defaults.update(overrides)
    return CSPPosition(**defaults)


class TestMarkToMarket:

    def test_updates_from_chain(self):
        mtm = CSPMarkToMarket()
        pos = _make_position(strike=150.0, expiry=date(2026, 7, 18))
        chain = _make_chain([
            {"strike": 150.0, "expiry": date(2026, 7, 18), "bid": 3.00, "ask": 3.20,
             "mid_price": 3.10, "delta": -0.35, "days_to_expiry": 15},
        ])
        current_date = date(2026, 7, 3)

        found = mtm.update_position(pos, chain, current_date)

        assert found is True
        assert pos.current_price == pytest.approx(3.10)
        assert pos.current_delta == pytest.approx(-0.35)
        assert pos.current_dte == 15

    def test_no_match_returns_false(self):
        mtm = CSPMarkToMarket()
        pos = _make_position(strike=150.0, expiry=date(2026, 7, 18))
        chain = _make_chain([
            {"strike": 145.0, "expiry": date(2026, 8, 15)},
        ])
        current_date = date(2026, 7, 3)

        found = mtm.update_position(pos, chain, current_date)

        assert found is False

    def test_empty_chain_returns_false(self):
        mtm = CSPMarkToMarket()
        pos = _make_position()
        chain = pd.DataFrame()
        current_date = date(2026, 7, 3)

        found = mtm.update_position(pos, chain, current_date)

        assert found is False


class TestBlackScholesFallback:

    def test_bs_put_price_otm(self):
        price = CSPMarkToMarket.bs_put_price(
            spot=160.0, strike=150.0, time_to_expiry_years=28 / 365,
            volatility=0.30,
        )
        assert price > 0
        assert price < 150.0

    def test_bs_put_price_deep_itm(self):
        intrinsic = 150.0 - 100.0
        price = CSPMarkToMarket.bs_put_price(
            spot=100.0, strike=150.0, time_to_expiry_years=1 / 365,
            volatility=0.30,
        )
        assert price >= intrinsic * 0.95

    def test_bs_put_price_zero_time_returns_intrinsic(self):
        price = CSPMarkToMarket.bs_put_price(
            spot=145.0, strike=150.0, time_to_expiry_years=0.0,
            volatility=0.30,
        )
        assert price == pytest.approx(5.0, abs=0.01)

    def test_estimate_price_uses_bs_fallback(self):
        mtm = CSPMarkToMarket()
        pos = _make_position(strike=150.0, expiry=date(2026, 7, 18))
        current_date = date(2026, 7, 3)

        price = mtm.estimate_price(
            position=pos,
            current_date=current_date,
            underlying_price=160.0,
            last_known_iv=0.30,
            risk_free_rate=0.05,
        )

        assert price > 0
