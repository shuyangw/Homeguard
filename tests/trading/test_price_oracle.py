"""Unit tests for `src.trading.price_oracle`.

Covers:
- PricePoint dataclass
- StreamingPriceProvider (bar present / missing / stale / zero close / not connected)
- BrokerQuotePriceProvider (last / mid / ask-only / missing / broker raises)
- BrokerCryptoQuotePriceProvider (parallel coverage to BrokerQuotePriceProvider)
- BrokerPortfolioPriceProvider (cache TTL / miss / refresh / broker raises)
- PriceOracle.get_live_price (fallback chain / all-None / provider raises)
- PriceOracle.enrich_position (happy path / oracle None / qty=0 / avg=0 / preserves cost basis)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.trading.price_oracle import (
    PricePoint,
    PriceOracle,
    StreamingPriceProvider,
    BrokerQuotePriceProvider,
    BrokerCryptoQuotePriceProvider,
    BrokerPortfolioPriceProvider,
)


# ============================================================================
# PricePoint
# ============================================================================

class TestPricePoint:
    def test_construction(self):
        pp = PricePoint(price=174.45, source="streaming", age_seconds=2.0)
        assert pp.price == 174.45
        assert pp.source == "streaming"
        assert pp.age_seconds == 2.0

    def test_age_seconds_optional(self):
        pp = PricePoint(price=174.45, source="broker_quote")
        assert pp.age_seconds is None

    def test_frozen(self):
        pp = PricePoint(price=174.45, source="streaming")
        with pytest.raises(Exception):
            pp.price = 200.0  # type: ignore[misc]


# ============================================================================
# Helpers
# ============================================================================

def _bar(close: float, age_seconds: float = 5.0):
    """Build a SimpleNamespace shaped like `Bar` (close + timestamp)."""
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return SimpleNamespace(close=close, timestamp=ts)


class _FakeStreaming:
    """Minimal duck-typed stand-in for StreamingProviderInterface."""
    def __init__(self, bars=None, raises=False, connected=True):
        self._bars = bars or {}
        self._raises = raises
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected

    def get_bar(self, symbol):
        if self._raises:
            raise RuntimeError("stream borked")
        return self._bars.get(symbol)


class _FakeBroker:
    def __init__(self, quotes=None, positions=None, raises_on_quote=False, raises_on_positions=False):
        self._quotes = quotes or {}
        self._positions = positions or []
        self._raises_on_quote = raises_on_quote
        self._raises_on_positions = raises_on_positions

    def get_latest_quote(self, symbol):
        if self._raises_on_quote:
            raise RuntimeError("quote borked")
        return self._quotes.get(symbol)

    def get_crypto_quote(self, symbol):
        return self.get_latest_quote(symbol)

    def get_stock_positions(self):
        if self._raises_on_positions:
            raise RuntimeError("positions borked")
        return self._positions


# ============================================================================
# StreamingPriceProvider
# ============================================================================

class TestStreamingPriceProvider:
    def test_bar_present_returns_close(self):
        s = _FakeStreaming(bars={"AAPL": _bar(174.45, age_seconds=2.0)})
        p = StreamingPriceProvider(streaming=s)
        out = p.get_live_price("AAPL")
        assert out is not None
        assert out.price == pytest.approx(174.45)
        assert out.source == "streaming"
        assert out.age_seconds is not None and out.age_seconds < 30

    def test_bar_missing_returns_none(self):
        p = StreamingPriceProvider(streaming=_FakeStreaming(bars={}))
        assert p.get_live_price("AAPL") is None

    def test_bar_close_zero_returns_none(self):
        p = StreamingPriceProvider(streaming=_FakeStreaming(bars={"AAPL": _bar(0.0)}))
        assert p.get_live_price("AAPL") is None

    def test_bar_close_none_returns_none(self):
        bar = SimpleNamespace(close=None, timestamp=datetime.now(timezone.utc))
        p = StreamingPriceProvider(streaming=_FakeStreaming(bars={"AAPL": bar}))
        assert p.get_live_price("AAPL") is None

    def test_bar_stale_returns_none(self):
        # max_age 60s, bar 120s old
        s = _FakeStreaming(bars={"AAPL": _bar(174.45, age_seconds=120)})
        p = StreamingPriceProvider(streaming=s, max_age_seconds=60)
        assert p.get_live_price("AAPL") is None

    def test_streaming_none_returns_none(self):
        p = StreamingPriceProvider(streaming=None)
        assert p.get_live_price("AAPL") is None

    def test_streaming_disconnected_returns_none(self):
        s = _FakeStreaming(bars={"AAPL": _bar(174.45)}, connected=False)
        p = StreamingPriceProvider(streaming=s)
        assert p.get_live_price("AAPL") is None

    def test_streaming_raises_returns_none(self):
        s = _FakeStreaming(raises=True)
        p = StreamingPriceProvider(streaming=s)
        # Should swallow the exception, not propagate
        assert p.get_live_price("AAPL") is None


# ============================================================================
# BrokerQuotePriceProvider
# ============================================================================

class TestBrokerQuotePriceProvider:
    def test_last_preferred(self):
        b = _FakeBroker(quotes={"AAPL": {"last": 100.0, "bid": 99.0, "ask": 101.0}})
        p = BrokerQuotePriceProvider(broker=b)
        out = p.get_live_price("AAPL")
        assert out is not None
        assert out.price == 100.0
        assert out.source == "broker_quote"

    def test_mid_used_when_no_last(self):
        b = _FakeBroker(quotes={"AAPL": {"bid": 99.0, "ask": 101.0}})
        p = BrokerQuotePriceProvider(broker=b)
        out = p.get_live_price("AAPL")
        assert out is not None
        assert out.price == pytest.approx(100.0)

    def test_ask_only(self):
        b = _FakeBroker(quotes={"AAPL": {"ask": 101.0}})
        p = BrokerQuotePriceProvider(broker=b)
        out = p.get_live_price("AAPL")
        assert out is not None
        assert out.price == 101.0

    def test_quote_missing_returns_none(self):
        b = _FakeBroker(quotes={})
        p = BrokerQuotePriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None

    def test_quote_all_zero_returns_none(self):
        b = _FakeBroker(quotes={"AAPL": {"last": 0.0, "bid": 0.0, "ask": 0.0}})
        p = BrokerQuotePriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None

    def test_broker_raises_returns_none(self):
        b = _FakeBroker(raises_on_quote=True)
        p = BrokerQuotePriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None


class TestBrokerCryptoQuotePriceProvider:
    def test_last_preferred(self):
        b = _FakeBroker(quotes={"BTC/USD": {"last": 77000.0}})
        p = BrokerCryptoQuotePriceProvider(broker=b)
        out = p.get_live_price("BTC/USD")
        assert out is not None
        assert out.price == 77000.0
        assert out.source == "broker_crypto_quote"


# ============================================================================
# BrokerPortfolioPriceProvider
# ============================================================================

class TestBrokerPortfolioPriceProvider:
    def test_cache_populated_from_first_call(self):
        b = _FakeBroker(positions=[{"symbol": "AAPL", "current_price": 174.45}])
        p = BrokerPortfolioPriceProvider(broker=b, cache_ttl_seconds=10)
        out = p.get_live_price("AAPL")
        assert out is not None
        assert out.price == 174.45
        assert out.source == "broker_portfolio"

    def test_cache_amortizes_calls(self):
        # Track how many times get_stock_positions is called.
        calls = {"n": 0}
        positions = [{"symbol": "AAPL", "current_price": 174.45}]

        class _CountingBroker:
            def get_stock_positions(self):
                calls["n"] += 1
                return positions

        p = BrokerPortfolioPriceProvider(broker=_CountingBroker(), cache_ttl_seconds=10)
        # 5 calls for the same symbol within TTL -> 1 broker hit
        for _ in range(5):
            p.get_live_price("AAPL")
        assert calls["n"] == 1

    def test_cache_expires_after_ttl(self):
        calls = {"n": 0}

        class _CountingBroker:
            def get_stock_positions(self):
                calls["n"] += 1
                return [{"symbol": "AAPL", "current_price": 174.45}]

        p = BrokerPortfolioPriceProvider(broker=_CountingBroker(), cache_ttl_seconds=0.01)
        p.get_live_price("AAPL")
        time.sleep(0.05)
        p.get_live_price("AAPL")
        assert calls["n"] == 2

    def test_symbol_missing_returns_none(self):
        b = _FakeBroker(positions=[{"symbol": "MSFT", "current_price": 410.0}])
        p = BrokerPortfolioPriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None

    def test_current_price_zero_returns_none(self):
        b = _FakeBroker(positions=[{"symbol": "AAPL", "current_price": 0}])
        p = BrokerPortfolioPriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None

    def test_current_price_none_returns_none(self):
        b = _FakeBroker(positions=[{"symbol": "AAPL", "current_price": None}])
        p = BrokerPortfolioPriceProvider(broker=b)
        assert p.get_live_price("AAPL") is None

    def test_broker_raises_returns_none_and_keeps_cache(self):
        # First call succeeds, second call (after TTL expiry) raises -> cache
        # stays from first call so subsequent gets within TTL still work, but
        # immediately after expiry the cache wasn't refreshed -> stays as last
        # known good. Verify _refresh swallows exception.
        positions = [{"symbol": "AAPL", "current_price": 174.45}]
        broker = SimpleNamespace(_count=0)

        def get_stock_positions():
            broker._count += 1
            if broker._count == 1:
                return positions
            raise RuntimeError("boom")

        broker.get_stock_positions = get_stock_positions  # type: ignore[attr-defined]

        p = BrokerPortfolioPriceProvider(broker=broker, cache_ttl_seconds=0.01)
        assert p.get_live_price("AAPL").price == 174.45
        time.sleep(0.05)
        # Second call: TTL expired, broker raises -> cache stays from before
        out = p.get_live_price("AAPL")
        assert out is not None and out.price == 174.45  # cached value preserved


# ============================================================================
# PriceOracle.get_live_price
# ============================================================================

class _DummyProvider:
    def __init__(self, return_value, raises=False, source="dummy"):
        self._return = return_value
        self._raises = raises
        self._source = source

    def get_live_price(self, symbol):
        if self._raises:
            raise RuntimeError("provider boom")
        return self._return


class TestPriceOracleFallbackChain:
    def test_first_provider_wins(self):
        oracle = PriceOracle(providers=[
            _DummyProvider(PricePoint(100.0, "first")),
            _DummyProvider(PricePoint(200.0, "second")),
        ])
        out = oracle.get_live_price("AAPL")
        assert out is not None and out.price == 100.0 and out.source == "first"

    def test_falls_through_when_first_returns_none(self):
        oracle = PriceOracle(providers=[
            _DummyProvider(None),
            _DummyProvider(PricePoint(200.0, "second")),
        ])
        out = oracle.get_live_price("AAPL")
        assert out is not None and out.price == 200.0 and out.source == "second"

    def test_all_none_returns_none(self):
        oracle = PriceOracle(providers=[
            _DummyProvider(None),
            _DummyProvider(None),
        ])
        assert oracle.get_live_price("AAPL") is None

    def test_provider_raise_falls_through(self):
        oracle = PriceOracle(providers=[
            _DummyProvider(None, raises=True),
            _DummyProvider(PricePoint(200.0, "second")),
        ])
        out = oracle.get_live_price("AAPL")
        assert out is not None and out.price == 200.0

    def test_empty_provider_list_returns_none(self):
        oracle = PriceOracle(providers=[])
        assert oracle.get_live_price("AAPL") is None

    def test_none_in_provider_list_skipped(self):
        oracle = PriceOracle(providers=[
            None,  # type: ignore[list-item]
            _DummyProvider(PricePoint(200.0, "second")),
        ])
        out = oracle.get_live_price("AAPL")
        assert out is not None and out.price == 200.0

    def test_empty_symbol_returns_none(self):
        oracle = PriceOracle(providers=[_DummyProvider(PricePoint(100.0, "first"))])
        assert oracle.get_live_price("") is None


# ============================================================================
# PriceOracle.enrich_position
# ============================================================================

class TestPriceOracleEnrich:
    def _oracle(self, price=None, source="streaming"):
        result = PricePoint(price, source) if price is not None else None
        return PriceOracle(providers=[_DummyProvider(result)])

    def test_happy_path_overwrites_pnl(self):
        oracle = self._oracle(price=180.00)
        pos = {
            "symbol": "AAPL",
            "quantity": 10,
            "avg_entry_price": 174.00,
            "current_price": 999.0,  # stale broker value -- should be overwritten
            "unrealized_pnl": -50.0,  # also stale
        }
        out = oracle.enrich_position(pos)
        assert out["current_price"] == 180.00
        assert out["market_value"] == pytest.approx(1800.0)
        assert out["unrealized_pnl"] == pytest.approx(60.0)  # (180 - 174) * 10
        assert out["unrealized_pnl_pct"] == pytest.approx((180 - 174) / 174 * 100.0)
        assert out["_price_source"] == "streaming"
        # cost basis preserved
        assert out["avg_entry_price"] == 174.00
        assert out["quantity"] == 10

    def test_oracle_returns_none_preserves_original(self):
        oracle = self._oracle(price=None)
        pos = {
            "symbol": "AAPL",
            "quantity": 10,
            "avg_entry_price": 174.00,
            "current_price": 999.0,
            "unrealized_pnl": -50.0,
        }
        out = oracle.enrich_position(pos)
        assert out["current_price"] == 999.0  # untouched
        assert out["unrealized_pnl"] == -50.0
        assert "_price_source" not in out

    def test_qty_zero_preserves_original(self):
        oracle = self._oracle(price=180.00)
        pos = {"symbol": "AAPL", "quantity": 0, "avg_entry_price": 174.00, "current_price": 999.0}
        out = oracle.enrich_position(pos)
        assert out["current_price"] == 999.0  # untouched -- nothing to mark

    def test_avg_zero_preserves_original(self):
        oracle = self._oracle(price=180.00)
        pos = {"symbol": "AAPL", "quantity": 10, "avg_entry_price": 0, "current_price": 999.0}
        out = oracle.enrich_position(pos)
        assert out["current_price"] == 999.0

    def test_no_symbol_preserves_original(self):
        oracle = self._oracle(price=180.00)
        pos = {"symbol": None, "quantity": 10, "avg_entry_price": 174.00}
        out = oracle.enrich_position(pos)
        assert out.get("current_price") is None or "current_price" not in out

    def test_returns_new_dict_doesnt_mutate_input(self):
        oracle = self._oracle(price=180.00)
        pos = {"symbol": "AAPL", "quantity": 10, "avg_entry_price": 174.00, "current_price": 999.0}
        out = oracle.enrich_position(pos)
        assert out is not pos
        assert pos["current_price"] == 999.0  # original unchanged

    def test_enrich_positions_vectorized(self):
        oracle = PriceOracle(providers=[
            _DummyProvider(PricePoint(180.0, "streaming")),
        ])
        positions = [
            {"symbol": "AAPL", "quantity": 10, "avg_entry_price": 174.0},
            {"symbol": "MSFT", "quantity": 5, "avg_entry_price": 400.0},
        ]
        out = oracle.enrich_positions(positions)
        assert len(out) == 2
        # both get the same dummy price (the dummy provider returns it for any symbol)
        assert out[0]["unrealized_pnl"] == pytest.approx((180 - 174) * 10)
        assert out[1]["unrealized_pnl"] == pytest.approx((180 - 400) * 5)


# ============================================================================
# Integration: full chain
# ============================================================================

class TestFullChain:
    def test_streaming_to_broker_quote_to_portfolio_fallback(self):
        # AAPL: streaming has it -> use streaming
        # MSFT: streaming missing, broker quote has it -> use broker quote
        # GOOG: both miss, broker portfolio has it -> use portfolio
        # NFLX: nothing has it -> None

        streaming = _FakeStreaming(bars={"AAPL": _bar(174.45)})
        broker = _FakeBroker(
            quotes={"MSFT": {"last": 410.0}},
            positions=[{"symbol": "GOOG", "current_price": 165.0}],
        )
        oracle = PriceOracle(providers=[
            StreamingPriceProvider(streaming=streaming),
            BrokerQuotePriceProvider(broker=broker),
            BrokerPortfolioPriceProvider(broker=broker),
        ])

        a = oracle.get_live_price("AAPL")
        assert a is not None and a.source == "streaming" and a.price == pytest.approx(174.45)

        m = oracle.get_live_price("MSFT")
        assert m is not None and m.source == "broker_quote" and m.price == 410.0

        g = oracle.get_live_price("GOOG")
        assert g is not None and g.source == "broker_portfolio" and g.price == 165.0

        n = oracle.get_live_price("NFLX")
        assert n is None
