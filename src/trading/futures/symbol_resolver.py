"""FuturesSymbolResolver: strategy intent -> broker order params.

Strategies generate signals on continuous (.v.0) series; live execution
needs the actual contract symbol for the day. This resolver is the
mapping boundary. Strategies pass continuous form ('ES.v.0') and get
back a ResolvedOrder with the contract-specific raw_symbol ('ESM4')
plus the contract_month.

See docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md
Section 2.2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.data.continuous_contract_loader import ContinuousContractDataLoader
from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce


_MONTH_CODES = "FGHJKMNQUVXZ"  # F=Jan ... Z=Dec
_CONTINUOUS_INTENT_RE = re.compile(r"^[A-Z0-9]+\.v\.\d+$")


class InvalidIntentError(ValueError):
    """Raised when strategy_intent is not in continuous (.v.0) form."""


@dataclass(frozen=True)
class ContractResolution:
    """The contract to trade today for a continuous symbol intent."""
    symbol_root: str
    contract_month: str   # YYYYMM
    raw_symbol: str       # e.g. ESM4


@dataclass(frozen=True)
class ResolvedOrder:
    """A futures order ready for broker submission."""
    strategy_intent: str       # original continuous symbol, e.g. ES.v.0
    symbol_root: str
    contract_month: str        # YYYYMM
    raw_symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None
    stop_price: float | None
    time_in_force: TimeInForce
    strategy: str              # which strategy emitted this order
    as_of: date
    expiration_date: date | None = None  # populated when definitions_loader is set


def _raw_symbol_to_contract_month(raw_symbol: str, root: str) -> str:
    """ESM4 + root=ES -> '202406'. Single-digit CME year decoded via current
    year's tens digit (2024 -> '2', so M4 -> 2024-06)."""
    suffix = raw_symbol[len(root):]
    if len(suffix) < 2:
        raise ValueError(f"cannot parse contract month from {raw_symbol}")
    month_letter = suffix[0]
    year_digit = int(suffix[1:])
    month_num = _MONTH_CODES.index(month_letter) + 1
    # Decode year: current decade tens-digit + the single year-digit.
    # If that gives a year more than 5 in the past, assume next decade.
    today = date.today()
    decade_base = (today.year // 10) * 10
    candidate = decade_base + year_digit
    if candidate < today.year - 5:
        candidate += 10
    return f"{candidate:04d}{month_num:02d}"


class FuturesSymbolResolver:
    """Resolves continuous symbol intents to per-contract order params."""

    def __init__(
        self,
        definitions_loader: FuturesDefinitionsLoader | None = None,
    ) -> None:
        self._loader = ContinuousContractDataLoader()
        self._definitions = definitions_loader
        self._cache: dict[tuple[str, date], ContractResolution] = {}

    def resolve_active_contract(self, symbol_root: str, as_of: date) -> ContractResolution:
        """Return today's active contract for the given root, cached."""
        key = (symbol_root, as_of)
        if key in self._cache:
            return self._cache[key]
        active_df = self._loader._active_contract_per_day(symbol_root, as_of, as_of)
        if active_df.is_empty():
            raise ValueError(f"no active contract for {symbol_root} on {as_of}")
        raw_symbol = active_df.row(0, named=True)["active"]
        cm = _raw_symbol_to_contract_month(raw_symbol, symbol_root)
        res = ContractResolution(
            symbol_root=symbol_root,
            contract_month=cm,
            raw_symbol=raw_symbol,
        )
        self._cache[key] = res
        return res

    def resolve_for_order(
        self,
        strategy_intent: str,
        side: OrderSide,
        quantity: int,
        as_of: date,
        strategy: str,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> ResolvedOrder:
        """Top-level: turn a continuous intent into a ready-to-submit order.

        Raises InvalidIntentError if strategy_intent isn't in continuous (.v.0)
        form. Forces discipline at the broker boundary -- raw symbols must
        come through this resolver, never bypass it.
        """
        if not _CONTINUOUS_INTENT_RE.match(strategy_intent):
            raise InvalidIntentError(
                f"strategy_intent must be in continuous form like 'ES.v.0', "
                f"got: {strategy_intent!r}"
            )
        root = strategy_intent.split(".", 1)[0]
        resolution = self.resolve_active_contract(root, as_of)
        expiration_date: date | None = None
        if self._definitions is not None:
            expiration_date = self._definitions.get_expiration(
                resolution.raw_symbol, root, as_of,
            )
        return ResolvedOrder(
            strategy_intent=strategy_intent,
            symbol_root=root,
            contract_month=resolution.contract_month,
            raw_symbol=resolution.raw_symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy=strategy,
            as_of=as_of,
            expiration_date=expiration_date,
        )
