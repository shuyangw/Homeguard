"""Per-asset-class cost models per docs/methodology/backtesting.md Section 4.

Every backtest must use these models -- a strategy that looks good before
costs is not a strategy. Replaces ad-hoc cost values scattered through
src/strategies/ and src/backtesting/engine/.

Public API:
    round_trip_cost_bps(asset_class, tier, ...) -> float
    equities_market_impact_bps(order_shares, daily_volume, daily_vol_pct, eta=0.5) -> float
    EQUITY_TIER_BPS, FUTURES_TICK_TABLE, FX_PIP_TIERS, CRYPTO_TIER_BPS, OPTIONS_ALPHA_TABLE
"""
from src.backtesting.costs.equities import (
    EQUITY_TIER_BPS,
    equities_market_impact_bps,
    equities_round_trip_bps,
)
from src.backtesting.costs.futures import FUTURES_PER_SIDE_USD, futures_round_trip_usd
from src.backtesting.costs.fx import FX_PIP_TIERS, fx_round_trip_pips
from src.backtesting.costs.crypto import CRYPTO_TIER_BPS, crypto_round_trip_bps
from src.backtesting.costs.options import (
    OPTIONS_ALPHA_TABLE,
    options_slippage_per_contract,
)

__all__ = [
    "EQUITY_TIER_BPS",
    "FUTURES_PER_SIDE_USD",
    "FX_PIP_TIERS",
    "CRYPTO_TIER_BPS",
    "OPTIONS_ALPHA_TABLE",
    "equities_round_trip_bps",
    "equities_market_impact_bps",
    "futures_round_trip_usd",
    "fx_round_trip_pips",
    "crypto_round_trip_bps",
    "options_slippage_per_contract",
]
