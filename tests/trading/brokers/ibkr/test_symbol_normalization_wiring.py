"""
Tests that the symbol normalization helpers are actually wired into
IBKR boundary methods.

`test_symbol_normalizer.py` proves the helpers work in isolation. These
tests prove they are called at the right points:
    - Outbound: `ContractResolver.resolve_stock` builds the IB Stock
      contract with space-format ('BRK B'), not dot-format ('BRK.B').
    - Inbound: `IBKRBroker._translate_stock_position` and
      `_translate_order` return dot-format symbols to the caller even
      when IB hands us space-format.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.trading.brokers.ibkr.contracts import ContractResolver
from src.trading.brokers.ibkr.ibkr_broker import IBKRBroker


def _build_resolver_with_passthrough_qualify():
    """Resolver whose `_qualify` echoes the input contract (with a conId)
    instead of calling IB. Lets us verify what Contract was constructed."""
    conn = MagicMock()
    resolver = ContractResolver(conn)

    def fake_qualify(contract):
        contract.conId = 12345
        return contract

    resolver._qualify = fake_qualify
    return resolver


class TestResolveStockNormalizesOutbound:
    def test_brk_b_becomes_space_in_contract(self):
        resolver = _build_resolver_with_passthrough_qualify()
        contract = resolver.resolve_stock("BRK.B")
        # The Stock contract sent to IB must use the space form
        assert contract.symbol == "BRK B"

    def test_bf_b_becomes_space_in_contract(self):
        resolver = _build_resolver_with_passthrough_qualify()
        contract = resolver.resolve_stock("BF.B")
        assert contract.symbol == "BF B"

    def test_regular_symbol_unchanged(self):
        resolver = _build_resolver_with_passthrough_qualify()
        contract = resolver.resolve_stock("AAPL")
        assert contract.symbol == "AAPL"


class TestTranslateMethodsNormalizeInbound:
    """The translate methods sit at the IB -> strategy boundary."""

    def _fake_position(self, ib_symbol: str, qty: int = 10, avg_cost: float = 450.0):
        """Build a minimal object shaped like an ib_async Position."""
        return SimpleNamespace(
            contract=SimpleNamespace(symbol=ib_symbol, secType="STK"),
            position=qty,
            avgCost=avg_cost,
        )

    def _fake_trade(self, ib_symbol: str):
        order = SimpleNamespace(
            orderId=42, totalQuantity=10, action="BUY",
            orderType="MKT", lmtPrice=0, auxPrice=0,
        )
        status = SimpleNamespace(
            status="Filled", filled=10, avgFillPrice=450.0,
        )
        return SimpleNamespace(
            contract=SimpleNamespace(symbol=ib_symbol, secType="STK"),
            order=order,
            orderStatus=status,
        )

    def test_stock_position_symbol_returned_in_dot_format(self):
        broker = IBKRBroker()
        pos = self._fake_position("BRK B")
        result = broker._translate_stock_position(pos)
        assert result["symbol"] == "BRK.B"

    def test_stock_position_regular_symbol_unchanged(self):
        broker = IBKRBroker()
        pos = self._fake_position("AAPL")
        result = broker._translate_stock_position(pos)
        assert result["symbol"] == "AAPL"

    def test_order_symbol_returned_in_dot_format(self):
        broker = IBKRBroker()
        trade = self._fake_trade("BF B")
        result = broker._translate_order(trade)
        assert result["symbol"] == "BF.B"

    def test_order_regular_symbol_unchanged(self):
        broker = IBKRBroker()
        trade = self._fake_trade("MSFT")
        result = broker._translate_order(trade)
        assert result["symbol"] == "MSFT"
