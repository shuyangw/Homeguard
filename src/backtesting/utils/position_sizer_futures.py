"""Contract-based futures position sizer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Static specification of a futures contract for sizing math."""
    symbol: str
    multiplier: float  # USD per point
    tick_size: float
    tick_value: float
    max_contracts: int


class FuturesPositionSizer:
    """Computes integer contract counts for vol-targeted sizing."""

    def size_position(
        self,
        symbol: str,
        vol_target_dollars: float,
        current_vol: float,
        contract_specs: ContractSpec,
        underlying_price: float,
    ) -> int:
        """Return integer number of contracts targeting `vol_target_dollars`
        annualized.

        Formula:
            vol_per_contract = multiplier * underlying_price * current_vol
            contracts = vol_target_dollars / vol_per_contract
        Returns 0 when current_vol <= 0 or vol_target_dollars <= 0.
        Hard-capped at contract_specs.max_contracts. Always returns integer >= 0.
        """
        if current_vol <= 0 or vol_target_dollars <= 0:
            return 0
        vol_per_contract = contract_specs.multiplier * underlying_price * current_vol
        if vol_per_contract <= 0:
            return 0
        raw = vol_target_dollars / vol_per_contract
        n = int(round(raw))
        if n < 0:
            n = 0
        return min(n, contract_specs.max_contracts)
