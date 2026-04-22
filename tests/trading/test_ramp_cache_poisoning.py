"""
Regression tests for the disk-cache poisoning bug discovered on 2026-04-22.

Root cause:
    v2 runner called preload_historical_data() at 09:31 ET (1 min after
    market open). Alpaca returned partial/missing daily bars for most
    symbols because the trading day had just started. The cache got
    persisted with a today-row that was 432/503 NaN (only 14% valid).
    At 15:55 ET, fetch_todays_closes() saw `last_date == today` and
    early-returned without refetching, so signal generation ran on the
    stale/incomplete data and placed 0 orders.

Fix (defense in depth):
    1. fetch_todays_closes validates today-row completeness before
       trusting it. <80% valid -> drop the row and fall through to
       streaming/REST path.
    2. preload_historical_data drops a today-row with <80% valid before
       persisting to disk so the cache stays clean for downstream use.

These tests lock in both behaviors.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.streaming.interface import StreamingProviderInterface
from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter


def _frozen_today():
    """A pandas Timestamp (midnight, UTC-naive) representing 'today' for RAMP."""
    from src.utils.timezone import tz
    now = tz.now()
    # tz.now() returns datetime.datetime. Convert to pd.Timestamp then normalize.
    return pd.Timestamp(now).normalize()


def _build_prices_df(symbols, dates, today_valid_pct=100):
    """Build a DataFrame whose last row has `today_valid_pct`% non-NaN values."""
    data = {s: [100.0 + i for i in range(len(dates))] for s in symbols}
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    if today_valid_pct < 100:
        # Set NaN on the last row for the bottom portion of symbols
        valid_count = int(round(len(symbols) * today_valid_pct / 100))
        nan_count = len(symbols) - valid_count
        for s in symbols[valid_count:]:
            df.loc[df.index[-1], s] = float('nan')
    return df


def _make_adapter(symbols, data_provider=None):
    adapter = RAMPLiveAdapter.__new__(RAMPLiveAdapter)
    adapter.symbols = symbols
    adapter.broker = MagicMock()
    adapter._data_provider = data_provider
    adapter._data_cache = None
    adapter._cache_date = None
    adapter._ramp_signals = MagicMock()
    adapter._ramp_signals.detect_regime.return_value = ('WEAK_BULL', 0.7)
    adapter.data_lookback_days = 300
    return adapter


def _alpaca_like_dp():
    """Data provider that satisfies both StreamingProviderInterface and
    get_historical_bars (AlpacaMarketData shape)."""
    class FakeAMD(StreamingProviderInterface):
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
    inst = FakeAMD()
    inst.get_bars = MagicMock(return_value=pd.DataFrame())
    inst.get_historical_bars = MagicMock()
    return inst


class TestFetchTodaysClosesValidityGate:
    """fetch_todays_closes must refetch when the cached today-row is mostly NaN."""

    def _cache_with_today_row(self, symbols, valid_pct):
        today = _frozen_today()
        yesterday = today - pd.Timedelta(days=1)
        prices = _build_prices_df(symbols, [yesterday, today], today_valid_pct=valid_pct)
        return {
            'prices': prices,
            'SPY': pd.DataFrame({'close': [430.0, 431.0]}, index=[yesterday, today]),
            'VIX': pd.DataFrame({'Close': [14.0, 15.0]}, index=[yesterday, today]),
        }

    def test_today_row_fully_valid_trusts_cache(self):
        symbols = [f'S{i}' for i in range(10)]
        dp = _alpaca_like_dp()
        adapter = _make_adapter(symbols, data_provider=dp)
        adapter._data_cache = self._cache_with_today_row(symbols, valid_pct=100)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # Fully valid cache -> no streaming call, no REST call
        dp.get_bars.assert_not_called()
        # SPY refetch uses REST; filter those out
        rest_universe_calls = [
            c for c in dp.get_historical_bars.call_args_list
            if c.kwargs.get('symbol') in symbols
        ]
        assert rest_universe_calls == []

    def test_today_row_85pct_valid_trusts_cache(self):
        """Just above 80% threshold -- trust."""
        symbols = [f'S{i}' for i in range(100)]
        dp = _alpaca_like_dp()
        adapter = _make_adapter(symbols, data_provider=dp)
        adapter._data_cache = self._cache_with_today_row(symbols, valid_pct=85)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        dp.get_bars.assert_not_called()

    def test_today_row_14pct_valid_refetches(self):
        """The actual bug: 71/503 ~= 14% valid, below threshold, must refetch."""
        symbols = [f'S{i}' for i in range(100)]
        dp = _alpaca_like_dp()
        dp.get_bars.return_value = pd.DataFrame()  # streaming empty
        dp.get_historical_bars.return_value = pd.DataFrame(
            {'close': [200.0]}, index=[_frozen_today()]
        )
        adapter = _make_adapter(symbols, data_provider=dp)
        adapter._data_cache = self._cache_with_today_row(symbols, valid_pct=14)

        ok = adapter.fetch_todays_closes()

        assert ok is True
        # Streaming attempted (data_provider is streaming-capable)
        assert dp.get_bars.call_count == len(symbols)
        # REST fallback also attempted (streaming was empty) -- symbols + SPY
        assert dp.get_historical_bars.call_count >= len(symbols)

    def test_today_row_drops_from_cache_before_refetch(self):
        """When today-row is invalid, it must be removed from the cache
        BEFORE refetch so that when refetch appends fresh data, we don't end up
        with two rows for today."""
        symbols = [f'S{i}' for i in range(20)]
        dp = _alpaca_like_dp()
        dp.get_bars.return_value = pd.DataFrame()
        dp.get_historical_bars.return_value = pd.DataFrame(
            {'close': [200.0]}, index=[_frozen_today()]
        )
        adapter = _make_adapter(symbols, data_provider=dp)
        adapter._data_cache = self._cache_with_today_row(symbols, valid_pct=25)
        original_prices_len = len(adapter._data_cache['prices'])

        adapter.fetch_todays_closes()

        # After refetch completes, cache has a TODAY row (new) -- but not
        # two today-rows (old incomplete + new). Count should stay at 2.
        assert len(adapter._data_cache['prices']) == original_prices_len


class TestPreloadHistoricalDataDropsIncompleteToday:
    """preload_historical_data must not persist a mostly-NaN today-row to disk."""

    def test_preload_drops_today_row_when_mostly_nan(self):
        """If preload runs at market open and today's bars are mostly empty,
        the disk cache should not persist a row for today at all."""
        symbols = [f'S{i}' for i in range(20)]
        today = _frozen_today()
        yesterday = today - pd.Timedelta(days=1)

        dp = _alpaca_like_dp()

        # Mock broker.get_historical_bars to return 2 bars for some symbols
        # (yesterday + today) and 1 bar for others (yesterday only) -- mimicking
        # Alpaca's behavior at market open.
        def fake_bars(symbol, start, end, timeframe='1D'):
            # SPY always has both days
            if symbol == 'SPY':
                return pd.DataFrame(
                    {'close': [430.0, 431.0]},
                    index=[yesterday, today]
                )
            # S0, S1, S2 have today bar; rest only yesterday -- mimicking
            # Alpaca returning partial data at 09:30 market open
            idx = int(symbol[1:])
            if idx < 3:
                return pd.DataFrame(
                    {'close': [100.0, 101.0]},
                    index=[yesterday, today]
                )
            return pd.DataFrame({'close': [100.0]}, index=[yesterday])
        dp.get_historical_bars.side_effect = fake_bars

        adapter = _make_adapter(symbols, data_provider=dp)
        adapter._ramp_signals.update_historical_data = MagicMock()
        adapter._fetch_vix_data = MagicMock(return_value=pd.DataFrame(
            {'Close': [14.0, 15.0]}, index=[yesterday, today]
        ))

        # Skip disk I/O
        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk', return_value=None), \
             patch('src.trading.adapters.ramp_live_adapter._save_cache_to_disk') as save_mock:
            adapter.preload_historical_data()

        # The persisted cache (argument to _save_cache_to_disk) should have
        # the today-row dropped -- last date should be yesterday.
        assert save_mock.called, "Disk cache should have been written"
        cache_arg = save_mock.call_args.args[0]
        persisted_prices = cache_arg['prices']
        last_cached_date = persisted_prices.index[-1]
        last_cached_date = last_cached_date.date() if hasattr(last_cached_date, 'date') else last_cached_date
        today_date = today.date() if hasattr(today, 'date') else today
        assert last_cached_date != today_date, (
            f"preload should drop today's incomplete row; persisted last_date={last_cached_date}"
        )
