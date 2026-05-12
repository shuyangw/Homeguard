"""Tests for the per-asset-class cost models."""
from __future__ import annotations

import math

import pytest

from src.backtesting.costs import (
    crypto_round_trip_bps,
    equities_market_impact_bps,
    equities_round_trip_bps,
    fx_round_trip_pips,
    futures_round_trip_usd,
    options_slippage_per_contract,
)


class TestEquities:
    def test_known_tiers(self) -> None:
        assert equities_round_trip_bps("large_cap_liquid") == pytest.approx(10.0)
        assert equities_round_trip_bps("mid_cap") == pytest.approx(22.5)
        assert equities_round_trip_bps("small_cap") == pytest.approx(125.0)

    def test_override(self) -> None:
        assert equities_round_trip_bps("large_cap_liquid", override_bps=3.0) == 3.0

    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown equity tier"):
            equities_round_trip_bps("bogus")  # type: ignore[arg-type]

    def test_market_impact_scales_sqrt(self) -> None:
        # Double the participation -> impact scales by sqrt(2).
        impact_1pct = equities_market_impact_bps(10_000, 1_000_000, 0.02)
        impact_4pct = equities_market_impact_bps(40_000, 1_000_000, 0.02)
        assert impact_4pct == pytest.approx(impact_1pct * 2.0)

    def test_market_impact_zero_for_bad_inputs(self) -> None:
        assert equities_market_impact_bps(100, 0, 0.02) == 0.0
        assert equities_market_impact_bps(0, 1_000_000, 0.02) == 0.0


class TestFutures:
    def test_es_regular_hours(self) -> None:
        # ES: per-side = 1.17 + 1.75 = 2.92. Tick = 12.50. Half-tick = 6.25.
        # Round-trip per contract = 2 * (2.92 + 6.25) = 18.34.
        assert futures_round_trip_usd("ES") == pytest.approx(18.34, abs=0.01)

    def test_off_hours_doubles_slippage(self) -> None:
        regular = futures_round_trip_usd("ES", regular_hours=True)
        off = futures_round_trip_usd("ES", regular_hours=False)
        # Off-hours adds another half-tick per side (12.50 total).
        assert off - regular == pytest.approx(12.50)

    def test_multiple_contracts(self) -> None:
        single = futures_round_trip_usd("CL", n_contracts=1)
        triple = futures_round_trip_usd("CL", n_contracts=3)
        assert triple == pytest.approx(3 * single)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown contract"):
            futures_round_trip_usd("XYZ")  # type: ignore[arg-type]


class TestFx:
    def test_major_london_ny(self) -> None:
        # major midpoint = 1.0 pip, london_ny mult = 1.0, x2 round-trip = 2.0.
        assert fx_round_trip_pips("major") == pytest.approx(2.0)

    def test_weekend_wider(self) -> None:
        london_ny = fx_round_trip_pips("major", "london_ny_overlap")
        weekend = fx_round_trip_pips("major", "weekend")
        assert weekend > london_ny

    def test_exotic_wider_than_major(self) -> None:
        assert fx_round_trip_pips("exotic") > fx_round_trip_pips("major")


class TestCrypto:
    def test_retail_base_120bps(self) -> None:
        # Retail (vol 0) taker on a major pair: (60 + 3) * 2 = 126 bps.
        # Methodology approximates as "120 bps retail" -- spread+fee.
        assert crypto_round_trip_bps("major", 0.0, "taker") == pytest.approx(126.0)

    def test_high_volume_lower_fee(self) -> None:
        retail = crypto_round_trip_bps("major", 0.0, "taker")
        institutional = crypto_round_trip_bps("major", 50_000_000, "taker")
        assert institutional < retail

    def test_maker_cheaper_than_taker(self) -> None:
        taker = crypto_round_trip_bps("major", 0.0, "taker")
        maker = crypto_round_trip_bps("major", 0.0, "maker")
        assert maker < taker

    def test_altcoin_higher(self) -> None:
        assert crypto_round_trip_bps("altcoin", 0.0) > crypto_round_trip_bps("major", 0.0)


class TestOptions:
    def test_alpha_table_orders_match_doc(self) -> None:
        from src.backtesting.costs import OPTIONS_ALPHA_TABLE
        # Section 4.5 ranks these tiers monotonically by liquidity.
        assert (
            OPTIONS_ALPHA_TABLE["very_liquid"]
            < OPTIONS_ALPHA_TABLE["liquid_etf"]
            < OPTIONS_ALPHA_TABLE["single_stock_atm"]
            < OPTIONS_ALPHA_TABLE["wings_illiquid"]
        )

    def test_buy_pays_above_mid(self) -> None:
        fill, slip = options_slippage_per_contract(
            bid=2.00, ask=2.10, side="buy", liquidity="single_stock_atm"
        )
        assert fill > 2.05  # above mid
        assert math.isclose(slip, 0.85 * 0.05, abs_tol=1e-9)

    def test_sell_pays_below_mid(self) -> None:
        fill, slip = options_slippage_per_contract(
            bid=2.00, ask=2.10, side="sell", liquidity="single_stock_atm"
        )
        assert fill < 2.05
        assert math.isclose(slip, 0.85 * 0.05, abs_tol=1e-9)

    def test_alpha_one_crosses_full_half_spread(self) -> None:
        fill, slip = options_slippage_per_contract(
            bid=2.00, ask=2.10, side="buy", override_alpha=1.0
        )
        assert fill == pytest.approx(2.10)  # paid the ask
        assert slip == pytest.approx(0.05)

    def test_inverted_market_raises(self) -> None:
        with pytest.raises(ValueError, match="ask"):
            options_slippage_per_contract(bid=2.10, ask=2.00, side="buy")

    def test_unknown_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side"):
            options_slippage_per_contract(bid=2.0, ask=2.1, side="hodl")  # type: ignore[arg-type]
