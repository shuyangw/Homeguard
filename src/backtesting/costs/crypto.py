"""Crypto cost model per docs/methodology/backtesting.md Section 4.4.

Coinbase Advanced Trade fee tiers by 30-day USD volume + per-pair spread.
Round-trip default for naive retail: 120 bps (60 taker + 60 taker).
"""
from __future__ import annotations

from typing import Literal, Optional

CryptoPairTier = Literal["major", "altcoin"]

# (taker_bps, maker_bps) by 30-day USD volume tier.
_VOLUME_TIERS_USD: list[tuple[float, float, float]] = [
    # (max_30d_volume_usd, taker_bps, maker_bps)
    (10_000, 60.0, 40.0),         # retail base
    (50_000, 40.0, 25.0),
    (100_000, 35.0, 15.0),
    (1_000_000, 25.0, 10.0),
    (15_000_000, 20.0, 8.0),
    (50_000_000, 18.0, 5.0),
    (100_000_000, 15.0, 0.0),
    (float("inf"), 10.0, 0.0),
]

# Per-pair spread on Coinbase (Section 4.4).
CRYPTO_TIER_BPS: dict[str, float] = {
    "major": 3.0,    # BTC/USD, ETH/USD: 1-5 bps midrange
    "altcoin": 30.0,  # Highly variable, 10-50 bps midrange
}


def _fees_for_volume(volume_30d_usd: float) -> tuple[float, float]:
    for ceiling, taker, maker in _VOLUME_TIERS_USD:
        if volume_30d_usd <= ceiling:
            return taker, maker
    return _VOLUME_TIERS_USD[-1][1], _VOLUME_TIERS_USD[-1][2]


def crypto_round_trip_bps(
    pair_tier: CryptoPairTier = "major",
    volume_30d_usd: float = 0.0,
    aggressor: Literal["taker", "maker"] = "taker",
    override_bps: Optional[float] = None,
) -> float:
    """Total round-trip cost in basis points.

    Args:
        pair_tier: "major" (BTC/ETH) or "altcoin" (per Section 4.4)
        volume_30d_usd: 30-day USD volume on the venue (drives the fee tier)
        aggressor: "taker" (default) or "maker"
        override_bps: if you have a measured value, pass it directly

    Returns:
        Round-trip cost in bps (fees in + fees out + per-pair spread).
    """
    if override_bps is not None:
        return float(override_bps)

    if pair_tier not in CRYPTO_TIER_BPS:
        raise ValueError(f"Unknown pair tier {pair_tier!r}. Choices: {list(CRYPTO_TIER_BPS)}")
    taker, maker = _fees_for_volume(volume_30d_usd)
    fee_per_side = taker if aggressor == "taker" else maker
    spread = CRYPTO_TIER_BPS[pair_tier]
    return 2.0 * (fee_per_side + spread)
