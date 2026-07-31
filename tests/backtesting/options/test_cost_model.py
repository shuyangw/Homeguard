"""Tests for M3 -- the options cost model (spec v2 Section 2, `cost_model_v2`).

The load-bearing test in this file is the UNIT RECONCILIATION: the chain quotes
fill conventions as a fraction of the FULL quoted width, while the pre-existing
`src/backtesting/costs/options.py` models `alpha` as a fraction of the
HALF-spread. 25% of width == alpha 0.50. Getting this wrong makes every cost
figure 2x off in one direction or the other.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtesting.costs.options import OPTIONS_ALPHA_TABLE, options_slippage_per_contract
from src.backtesting.options.cost_model import (
    ASSUMED_WIDTH_TABLE,
    COMBO_WIDTH_FRAC,
    SEC_S31_EFFECTIVE_DATE,
    SEC_S31_RATE_PER_DOLLAR,
    SINGLE_LEG_WIDTH_FRAC,
    STRESSED_COMBO_WIDTH_FRAC,
    CostBreakdown,
    FeeSchedule,
    Leg,
    Tier,
    abs_delta_bucket,
    alpha_from_width_fraction,
    cost,
    dte_bucket,
    moneyness_bucket,
    per_contract_fees,
    resolve_tier,
    vol_state_bucket,
    width_fraction_for,
    width_fraction_from_alpha,
)

SESSION = date(2024, 6, 3)
EXPIRY = date(2024, 7, 19)


def _leg(side="sell", bid=1.00, ask=1.06, qty=1, dte=46, delta=0.16, strike=520.0):
    return Leg(
        root="SPY",
        session_date=SESSION,
        expiry=EXPIRY,
        strike=strike,
        right="P",
        side=side,
        qty=qty,
        bid=bid,
        ask=ask,
        dte=dte,
        delta=delta,
        underlying_px=530.0,
    )


# --------------------------------------------------------------------------
# THE UNIT TRAP
# --------------------------------------------------------------------------

def test_25pct_of_width_is_alpha_one_half():
    assert alpha_from_width_fraction(0.25) == pytest.approx(0.50)
    assert width_fraction_from_alpha(0.50) == pytest.approx(0.25)


def test_width_fraction_alpha_roundtrip():
    for f in (0.03, 0.045, 0.06, 0.15, 0.20, 0.25, 0.55):
        assert width_fraction_from_alpha(alpha_from_width_fraction(f)) == pytest.approx(f)


def test_reconciliation_against_the_existing_alpha_module_with_worked_numbers():
    """bid 1.00 / ask 1.20 -> width 0.20, half-spread 0.10.

    25% of width = $0.05 per share = $5.00 per contract.
    The existing module with alpha = 0.50 must produce exactly that.
    """
    bid, ask = 1.00, 1.20
    fill, slip = options_slippage_per_contract(
        bid, ask, side="buy", override_alpha=alpha_from_width_fraction(0.25)
    )
    assert fill == pytest.approx(1.15)
    assert slip == pytest.approx(0.05)
    assert slip == pytest.approx(0.25 * (ask - bid))


def test_existing_alpha_table_translated_into_width_fractions():
    """A cross-check that the two conventions are stated, not conflated."""
    assert width_fraction_from_alpha(OPTIONS_ALPHA_TABLE["very_liquid"]) == pytest.approx(0.20)
    assert width_fraction_from_alpha(OPTIONS_ALPHA_TABLE["liquid_etf"]) == pytest.approx(0.30)
    assert width_fraction_from_alpha(OPTIONS_ALPHA_TABLE["wings_illiquid"]) == pytest.approx(0.55)


# --------------------------------------------------------------------------
# Registered fill fractions
# --------------------------------------------------------------------------

def test_registered_fill_fractions():
    assert SINGLE_LEG_WIDTH_FRAC == pytest.approx(0.25)
    assert COMBO_WIDTH_FRAC == (0.03, 0.06)
    assert STRESSED_COMBO_WIDTH_FRAC == (0.15, 0.25)


def test_single_leg_normal_pays_25pct_of_width():
    assert width_fraction_for(n_legs=1, tier=Tier.NORMAL, dte=46) == pytest.approx(0.25)


def test_native_combo_normal_pays_3_to_6pct_per_leg():
    f = width_fraction_for(n_legs=4, tier=Tier.NORMAL, dte=46)
    assert COMBO_WIDTH_FRAC[0] <= f <= COMBO_WIDTH_FRAC[1]


def test_five_leg_structure_is_not_a_native_combo():
    """The registered combo convention covers 2-4 legs only."""
    assert width_fraction_for(n_legs=5, tier=Tier.NORMAL, dte=46) == pytest.approx(0.25)


def test_stressed_tier_prices_combos_at_single_leg_grade():
    f = width_fraction_for(n_legs=4, tier=Tier.STRESSED, dte=46)
    assert STRESSED_COMBO_WIDTH_FRAC[0] <= f <= STRESSED_COMBO_WIDTH_FRAC[1]
    assert f > COMBO_WIDTH_FRAC[1]


def test_0dte_is_always_single_leg_grade():
    assert width_fraction_for(n_legs=4, tier=Tier.NORMAL, dte=0) == pytest.approx(
        SINGLE_LEG_WIDTH_FRAC
    )


def test_tier_resolution():
    assert resolve_tier(regime_state="STRONG_BULL") is Tier.NORMAL
    assert resolve_tier(regime_state="BEAR") is Tier.STRESSED
    assert resolve_tier(regime_state="UNPREDICTABLE") is Tier.STRESSED
    assert resolve_tier(regime_state="SIDEWAYS", rv_percentile=0.85) is Tier.STRESSED
    assert resolve_tier(regime_state="SIDEWAYS", rv_percentile=0.80) is Tier.NORMAL
    assert resolve_tier(regime_state="SIDEWAYS", event_window=True) is Tier.STRESSED


# --------------------------------------------------------------------------
# Fee stack
# --------------------------------------------------------------------------

def test_fee_components_buy_side():
    f = FeeSchedule()
    got = per_contract_fees(side="buy", session=SESSION, proceeds_usd=0.0, fees=f)
    expected = f.commission + f.occ_clearing + f.orf + f.cat
    assert got == pytest.approx(expected)


def test_taf_is_charged_on_sells_only():
    f = FeeSchedule()
    buy = per_contract_fees(side="buy", session=SESSION, proceeds_usd=0.0, fees=f)
    sell = per_contract_fees(side="sell", session=SESSION, proceeds_usd=0.0, fees=f)
    assert sell - buy == pytest.approx(f.taf_sell_only)


def test_sec_section31_starts_2026_04_04():
    f = FeeSchedule()
    before = per_contract_fees(
        side="sell", session=date(2026, 4, 3), proceeds_usd=100.0, fees=f
    )
    after = per_contract_fees(
        side="sell", session=date(2026, 4, 4), proceeds_usd=100.0, fees=f
    )
    assert SEC_S31_EFFECTIVE_DATE == date(2026, 4, 4)
    assert after - before == pytest.approx(100.0 * SEC_S31_RATE_PER_DOLLAR)
    assert SEC_S31_RATE_PER_DOLLAR == pytest.approx(20.60 / 1e6)


def test_all_in_one_way_fee_sits_in_the_registered_band_at_default_commission():
    f = FeeSchedule()
    for side in ("buy", "sell"):
        v = per_contract_fees(side=side, session=SESSION, proceeds_usd=0.0, fees=f)
        assert 0.42 <= v <= 0.72


# --------------------------------------------------------------------------
# cost() -- the registered signature
# --------------------------------------------------------------------------

def test_cost_signature_unpacks_to_three_values():
    entry, exit_, fees = cost([_leg()], regime_state="SIDEWAYS", width_source="assumed")
    assert entry > 0 and exit_ > 0 and fees > 0


def test_single_leg_entry_cost_is_width_fraction_times_width_times_100():
    br = cost([_leg(bid=1.00, ask=1.20)], regime_state="SIDEWAYS", width_source="quote")
    assert isinstance(br, CostBreakdown)
    assert br.entry_cost == pytest.approx(0.25 * 0.20 * 100.0)


def test_cost_scales_linearly_with_quantity():
    one = cost([_leg(qty=1)], regime_state="SIDEWAYS", width_source="quote")
    ten = cost([_leg(qty=10)], regime_state="SIDEWAYS", width_source="quote")
    assert ten.entry_cost == pytest.approx(10.0 * one.entry_cost)
    assert ten.fees == pytest.approx(10.0 * one.fees)


def test_cost_multiplier_reports_at_1x_and_plus_minus_50pct():
    base = cost([_leg()], regime_state="SIDEWAYS", width_source="assumed")
    hi = cost([_leg()], regime_state="SIDEWAYS", width_source="assumed", cost_multiplier=1.5)
    lo = cost([_leg()], regime_state="SIDEWAYS", width_source="assumed", cost_multiplier=0.5)
    assert hi.entry_cost == pytest.approx(1.5 * base.entry_cost)
    assert lo.entry_cost == pytest.approx(0.5 * base.entry_cost)
    assert hi.fees == pytest.approx(1.5 * base.fees)


def test_stressed_regime_costs_more_than_normal_for_a_combo():
    legs = [_leg(side="sell"), _leg(side="buy", strike=500.0)]
    normal = cost(legs, regime_state="STRONG_BULL", width_source="quote")
    stressed = cost(legs, regime_state="BEAR", width_source="quote")
    assert stressed.entry_cost > normal.entry_cost
    assert normal.tier is Tier.NORMAL and stressed.tier is Tier.STRESSED


def test_width_source_is_recorded_per_leg():
    br = cost([_leg()], regime_state="SIDEWAYS", width_source="assumed")
    assert br.width_source_by_leg == ["assumed"]


def test_census_is_used_when_the_cell_exists_and_assumed_otherwise():
    census = pd.DataFrame(
        [
            {
                "root": "SPY",
                "year": 2024,
                "moneyness_bucket": moneyness_bucket(520.0, 530.0),
                "dte_bucket": dte_bucket(46),
                "abs_delta_bucket": abs_delta_bucket(0.16),
                "vol_state_proxy": "vol_mid",
                "spread_abs_p50": 0.08,
                "n_sampled": 900,
            }
        ]
    )
    hit = cost(
        [_leg()],
        regime_state="SIDEWAYS",
        width_source="census",
        census=census,
        vol_state="vol_mid",
    )
    assert hit.width_source_by_leg == ["census"]
    assert hit.entry_cost == pytest.approx(0.25 * 0.08 * 100.0)

    miss = cost(
        [_leg(dte=200)],
        regime_state="SIDEWAYS",
        width_source="census",
        census=census,
        vol_state="vol_mid",
    )
    assert miss.width_source_by_leg == ["assumed"]


def test_census_cell_below_min_sample_falls_back_to_assumed():
    census = pd.DataFrame(
        [
            {
                "root": "SPY",
                "year": 2024,
                "moneyness_bucket": moneyness_bucket(520.0, 530.0),
                "dte_bucket": dte_bucket(46),
                "abs_delta_bucket": abs_delta_bucket(0.16),
                "vol_state_proxy": "vol_mid",
                "spread_abs_p50": 0.08,
                "n_sampled": 3,
            }
        ]
    )
    br = cost(
        [_leg()], regime_state="SIDEWAYS", width_source="census",
        census=census, vol_state="vol_mid",
    )
    assert br.width_source_by_leg == ["assumed"]


def test_assumed_table_has_the_registered_classes():
    for key in ("index_atm_30_45", "index_0dte_atm", "index_low_delta",
                "single_name_atm", "rank_50_100_atm", "leaps"):
        assert key in ASSUMED_WIDTH_TABLE


# --------------------------------------------------------------------------
# Bucket functions must agree with the census builder they index into
# --------------------------------------------------------------------------

def test_bucket_functions_agree_with_the_census_builder():
    from scripts.data.vbattery._sweep_lib import (
        ABS_DELTA_LABELS,
        DTE_LABELS,
        MONEYNESS_LABELS,
        abs_delta_code,
        dte_code,
        moneyness_code,
    )

    deltas = np.array([0.0, 0.01, 0.05, 0.1499, 0.15, 0.34, 0.35, 0.65, 0.66, 1.0, np.nan])
    want = abs_delta_code(deltas)
    for d, c in zip(deltas, want):
        assert abs_delta_bucket(float(d)) == ABS_DELTA_LABELS[int(c)]

    dtes = np.array([0, 1, 7, 8, 30, 31, 60, 61, 90, 91, 180, 181, 900], dtype=float)
    want = dte_code(dtes)
    for d, c in zip(dtes, want):
        assert dte_bucket(int(d)) == DTE_LABELS[int(c)]

    und = np.full(9, 100.0)
    strikes = np.array([80.0, 90.0, 94.0, 96.0, 100.0, 103.0, 106.0, 111.0, 150.0])
    want, _ = moneyness_code(strikes, und)
    for k, c in zip(strikes, want):
        assert moneyness_bucket(float(k), 100.0) == MONEYNESS_LABELS[int(c)]


def test_vol_state_bucket_uses_the_census_terciles():
    assert vol_state_bucket(0.10) == "vol_low"
    assert vol_state_bucket(0.50) == "vol_mid"
    assert vol_state_bucket(0.90) == "vol_high"
    assert vol_state_bucket(None) == "vol_unknown"
