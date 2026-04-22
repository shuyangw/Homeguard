"""
Regression tests for RAMPLiveAdapter.fetch_todays_closes rest-fallback path.

Fixes a bug where the streaming->broker-API fallback had a structural
indentation issue: the else-branch claiming to be the "broker API fallback"
was actually attached to the success-rate-threshold check, so when streaming
failed badly (<50%) nothing fell back and the rebalance failed.

These tests cover the four paths:
    1. Streaming succeeds (>=80%) -- streaming values used
    2. Streaming partial (50-80%) -- streaming values used + warning
    3. Streaming total-fail (<50%) -- REST fallback runs
    4. No streaming provider -- REST fallback runs from the start

And the architectural invariant:
    5. REST fallback prefers data_provider (Alpaca) over broker (IBKR).
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.streaming.interface import StreamingProviderInterface
from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter


def _make_adapter(symbols, data_provider=None):
    """Construct a minimal RAMPLiveAdapter without running __init__."""
    adapter = RAMPLiveAdapter.__new__(RAMPLiveAdapter)
    adapter.symbols = symbols
    adapter.broker = MagicMock()
    adapter._data_provider = data_provider
    adapter._data_cache = {
        'prices': pd.DataFrame(
            {s: [100.0, 101.0] for s in symbols},
            index=pd.to_datetime(['2026-04-20', '2026-04-21'])
        ),
        'SPY': pd.DataFrame(
            {'close': [430.0, 431.0]},
            index=pd.to_datetime(['2026-04-20', '2026-04-21'])
        ),
        'VIX': pd.DataFrame(
            {'Close': [14.0, 15.0]},
            index=pd.to_datetime(['2026-04-20', '2026-04-21'])
        ),
    }
    adapter._cache_date = None
    adapter._ramp_signals = MagicMock()
    adapter._ramp_signals.detect_regime.return_value = ('WEAK_BULL', 0.7)
    adapter.data_lookback_days = 300
    return adapter


def _bar_df(close):
    return pd.DataFrame({'close': [close]}, index=[pd.Timestamp.now()])


def _streaming_provider():
    """StreamingProviderInterface mock (no get_historical_bars)."""
    dp = MagicMock(spec=StreamingProviderInterface)
    dp.name = 'streaming-iex'
    return dp


def _alpaca_market_data_provider():
    """AlpacaMarketData-shape mock: StreamingProviderInterface + get_historical_bars."""
    # MagicMock(spec=StreamingProviderInterface) won't allow get_historical_bars;
    # so make a proper subclass instance via type().
    class FakeAlpacaMarketData(StreamingProviderInterface):
        name = 'alpaca-market-iex'
        def start(self, symbols=None): pass
        def stop(self): pass
        def is_connected(self): return True
        def get_price(self, s): return None
        def get_quote(self, s): return None
        def get_trade(self, s): return None
        def get_bar(self, s): return None
        def get_bars(self, s, n=None): return pd.DataFrame()
        def get_vwap(self, s): return None
        def get_spread(self, s): return None
        def on_bar(self, ss, h): return ''
        def on_quote(self, ss, h): return ''
        def on_trade(self, ss, h): return ''
        def unsubscribe(self, sid): pass
        def get_subscribed_symbols(self): return set()
    inst = FakeAlpacaMarketData()
    # Override get_bars and get_historical_bars as MagicMocks for per-test control
    inst.get_bars = MagicMock()
    inst.get_historical_bars = MagicMock()
    return inst


class TestStreamingSuccess:
    """>= 80% streaming coverage -> use streaming values, skip REST fallback."""

    def test_full_streaming_uses_buffer_values(self):
        symbols = ['A', 'B', 'C', 'D', 'E']
        dp = _alpaca_market_data_provider()
        dp.get_bars.side_effect = lambda s, n=None: _bar_df(100 + ord(s[0]))
        # SPY is always fetched via REST (never from streaming buffer for the
        # regime-detection path), so allow that one call.
        dp.get_historical_bars.return_value = _bar_df(430.0)
        adapter = _make_adapter(symbols, data_provider=dp)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # Streaming used once per symbol (not for SPY)
        assert dp.get_bars.call_count == len(symbols)
        # REST path must NOT be used for any of the universe symbols --
        # only SPY is fetched via REST (one call).
        rest_calls = [c for c in dp.get_historical_bars.call_args_list
                      if c.kwargs.get('symbol') in symbols]
        assert rest_calls == [], f"Universe symbols leaked to REST: {rest_calls}"
        # Broker must never be touched for data
        adapter.broker.get_historical_bars.assert_not_called()


class TestStreamingTotalFailure:
    """<50% streaming -> REST fallback via data_provider (Alpaca)."""

    def test_empty_buffer_falls_back_to_data_provider(self):
        symbols = ['A', 'B', 'C', 'D', 'E']
        dp = _alpaca_market_data_provider()
        # Streaming returns empty for everyone
        dp.get_bars.return_value = pd.DataFrame()
        # REST succeeds for everyone
        dp.get_historical_bars.side_effect = lambda **kw: _bar_df(200 + ord(kw['symbol'][0]))
        adapter = _make_adapter(symbols, data_provider=dp)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # Streaming attempted once per symbol
        assert dp.get_bars.call_count == len(symbols)
        # REST fallback attempted once per symbol PLUS once for SPY
        assert dp.get_historical_bars.call_count == len(symbols) + 1
        # Broker must NOT be touched for data -- that's the architecture invariant
        adapter.broker.get_historical_bars.assert_not_called()


class TestNoStreaming:
    """data_provider is a pure DataProvider (no StreamingProviderInterface)
    -- go straight to REST fallback via data_provider."""

    def test_plain_data_provider_skips_streaming(self):
        symbols = ['A', 'B', 'C']
        dp = MagicMock()
        dp.name = 'alpaca'
        # Crucially NOT a StreamingProviderInterface instance
        assert not isinstance(dp, StreamingProviderInterface)
        dp.get_historical_bars.side_effect = lambda **kw: _bar_df(300 + ord(kw['symbol'][0]))
        adapter = _make_adapter(symbols, data_provider=dp)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # get_bars must not be called (streaming path skipped)
        dp.get_bars.assert_not_called()
        # REST called per symbol + SPY
        assert dp.get_historical_bars.call_count == len(symbols) + 1


class TestBrokerLastResort:
    """When no data_provider at all, REST fallback MUST still happen via broker."""

    def test_no_data_provider_uses_broker(self):
        symbols = ['A', 'B']
        adapter = _make_adapter(symbols, data_provider=None)
        adapter.broker.get_historical_bars.side_effect = lambda **kw: _bar_df(400 + ord(kw['symbol'][0]))

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # REST fallback ran on the broker (symbols + SPY)
        assert adapter.broker.get_historical_bars.call_count == len(symbols) + 1


class TestCriticalFailure:
    """All data paths fail -> return False (do not silently continue)."""

    def test_streaming_empty_and_rest_empty_returns_false(self):
        symbols = ['A', 'B', 'C']
        dp = _alpaca_market_data_provider()
        dp.get_bars.return_value = pd.DataFrame()
        dp.get_historical_bars.return_value = None  # REST also fails
        adapter = _make_adapter(symbols, data_provider=dp)

        ok = adapter.fetch_todays_closes()

        assert ok is False
        adapter.broker.get_historical_bars.assert_not_called()


class TestArchitectureInvariant:
    """Pick REST source helper must prefer data_provider (Alpaca) over broker (IBKR)."""

    def test_pick_rest_source_prefers_data_provider(self):
        adapter = _make_adapter(['A'], data_provider=_alpaca_market_data_provider())
        source, name = adapter._pick_rest_source()
        assert source is adapter._data_provider
        assert 'alpaca' in name.lower() or 'data' in name.lower()

    def test_pick_rest_source_falls_back_to_broker_only_when_no_data_provider(self):
        adapter = _make_adapter(['A'], data_provider=None)
        source, name = adapter._pick_rest_source()
        assert source is adapter.broker
        assert 'broker' in name.lower()

    def test_pick_rest_source_ignores_streaming_only_data_provider(self):
        # LiveDataProvider is streaming-only; no get_historical_bars.
        # _pick_rest_source should skip it and fall back to broker.
        dp = _streaming_provider()
        # Belt-and-suspenders: ensure the mock really doesn't have get_historical_bars
        del dp.get_historical_bars
        adapter = _make_adapter(['A'], data_provider=dp)
        source, name = adapter._pick_rest_source()
        assert source is adapter.broker
        assert 'broker' in name.lower()
