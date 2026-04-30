"""Tests for MetricsRegistry - thread safety, metric types, prometheus format."""

import threading
import time

from src.monitoring.registry import MetricsRegistry


class TestGaugeOperations:
    """Test gauge set/get behavior."""

    def test_set_and_get_gauge(self):
        reg = MetricsRegistry(strategy='omr')
        reg.set_gauge('hg_portfolio_equity_usd', 100000.0, {'broker': 'alpaca'})
        value = reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'alpaca'})
        assert value == 100000.0

    def test_gauge_overwrites_previous(self):
        reg = MetricsRegistry(strategy='omr')
        reg.set_gauge('hg_portfolio_equity_usd', 100000.0, {'broker': 'alpaca'})
        reg.set_gauge('hg_portfolio_equity_usd', 99000.0, {'broker': 'alpaca'})
        value = reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'alpaca'})
        assert value == 99000.0

    def test_gauge_different_labels_independent(self):
        reg = MetricsRegistry(strategy='omr')
        reg.set_gauge('hg_portfolio_equity_usd', 100000.0, {'broker': 'alpaca'})
        reg.set_gauge('hg_portfolio_equity_usd', 50000.0, {'broker': 'ibkr'})
        assert reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'alpaca'}) == 100000.0
        assert reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'ibkr'}) == 50000.0

    def test_gauge_no_labels(self):
        reg = MetricsRegistry(strategy='omr')
        reg.set_gauge('hg_market_open', 1.0)
        assert reg.get_gauge('hg_market_open') == 1.0

    def test_gauge_missing_returns_none(self):
        reg = MetricsRegistry(strategy='omr')
        assert reg.get_gauge('nonexistent') is None


class TestCounterOperations:
    """Test counter increment behavior."""

    def test_increment_counter(self):
        reg = MetricsRegistry(strategy='omr')
        reg.inc_counter('hg_orders_submitted_total', {'strategy': 'omr', 'side': 'buy', 'broker': 'alpaca'})
        reg.inc_counter('hg_orders_submitted_total', {'strategy': 'omr', 'side': 'buy', 'broker': 'alpaca'})
        value = reg.get_counter('hg_orders_submitted_total', {'strategy': 'omr', 'side': 'buy', 'broker': 'alpaca'})
        assert value == 2

    def test_counter_starts_at_zero(self):
        reg = MetricsRegistry(strategy='omr')
        value = reg.get_counter('hg_orders_submitted_total', {'strategy': 'omr', 'side': 'buy', 'broker': 'alpaca'})
        assert value == 0

    def test_increment_by_amount(self):
        reg = MetricsRegistry(strategy='omr')
        reg.inc_counter('hg_orders_filled_total', {'strategy': 'omr', 'broker': 'alpaca'}, amount=5)
        assert reg.get_counter('hg_orders_filled_total', {'strategy': 'omr', 'broker': 'alpaca'}) == 5


class TestHistogramOperations:
    """Test histogram observe behavior."""

    def test_observe_histogram(self):
        reg = MetricsRegistry(strategy='omr')
        reg.observe_histogram('hg_fill_slippage_bps', 2.5, {'strategy': 'omr'})
        reg.observe_histogram('hg_fill_slippage_bps', 5.0, {'strategy': 'omr'})
        reg.observe_histogram('hg_fill_slippage_bps', 1.0, {'strategy': 'omr'})
        hist = reg.get_histogram('hg_fill_slippage_bps', {'strategy': 'omr'})
        assert hist['count'] == 3
        assert hist['sum'] == 8.5


class TestRemoveMetric:
    """Test metric removal (for position close cleanup)."""

    def test_remove_gauge_labels(self):
        reg = MetricsRegistry(strategy='omr')
        reg.set_gauge('hg_position_qty', 100.0, {'symbol': 'AAPL', 'strategy': 'omr'})
        reg.set_gauge('hg_position_qty', 50.0, {'symbol': 'MSFT', 'strategy': 'omr'})
        reg.remove_gauge('hg_position_qty', {'symbol': 'AAPL', 'strategy': 'omr'})
        assert reg.get_gauge('hg_position_qty', {'symbol': 'AAPL', 'strategy': 'omr'}) is None
        assert reg.get_gauge('hg_position_qty', {'symbol': 'MSFT', 'strategy': 'omr'}) == 50.0


class TestThreadSafety:
    """Test concurrent access to registry."""

    def test_concurrent_gauge_updates(self):
        reg = MetricsRegistry(strategy='omr')
        errors = []

        def writer(value):
            try:
                for _ in range(1000):
                    reg.set_gauge('hg_portfolio_equity_usd', value, {'broker': 'alpaca'})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final value should be one of the writer values (0-9)
        final = reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'alpaca'})
        assert final is not None

    def test_concurrent_counter_increments(self):
        reg = MetricsRegistry(strategy='omr')
        errors = []

        def incrementer():
            try:
                for _ in range(1000):
                    reg.inc_counter('hg_orders_filled_total', {'strategy': 'omr', 'broker': 'alpaca'})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=incrementer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert reg.get_counter('hg_orders_filled_total', {'strategy': 'omr', 'broker': 'alpaca'}) == 10000


class TestHighLevelUpdates:
    """Test convenience methods that map to low-level gauge/counter ops."""

    def test_update_portfolio(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_portfolio(
            equity_usd=100000.0,
            cash_usd=50000.0,
            buying_power_usd=150000.0,
            broker='alpaca'
        )
        assert reg.get_gauge('hg_portfolio_equity_usd', {'broker': 'alpaca'}) == 100000.0
        assert reg.get_gauge('hg_portfolio_cash_usd', {'broker': 'alpaca'}) == 50000.0
        assert reg.get_gauge('hg_portfolio_buying_power_usd', {'broker': 'alpaca'}) == 150000.0

    def test_update_drawdown(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_drawdown(5.2)
        assert reg.get_gauge('hg_portfolio_drawdown_pct') == 5.2

    def test_update_strategy_last_decision_timestamp_is_labeled(self):
        """Decision-age gauge must carry the strategy label so each strategy
        has its own series. Without the label, the lookup must miss."""
        reg = MetricsRegistry(strategy='ramp')
        reg.update_strategy_last_decision_timestamp(1700000000.5)
        assert reg.get_gauge(
            'hg_strategy_last_decision_timestamp', {'strategy': 'ramp'}
        ) == 1700000000.5
        assert reg.get_gauge('hg_strategy_last_decision_timestamp') is None

    def test_update_day_pnl(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_day_pnl(-150.0)
        assert reg.get_gauge('hg_portfolio_day_pnl_usd') == -150.0

    def test_update_strategy(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_strategy(
            realized_pnl=500.0,
            unrealized_pnl=-200.0,
            positions=3,
            capital_allocated=30000.0,
            last_signal_ts=1713400000.0
        )
        labels = {'strategy': 'omr'}
        assert reg.get_gauge('hg_strategy_realized_pnl_usd', labels) == 500.0
        assert reg.get_gauge('hg_strategy_unrealized_pnl_usd', labels) == -200.0
        assert reg.get_gauge('hg_strategy_positions_count', labels) == 3
        assert reg.get_gauge('hg_strategy_capital_allocated_usd', labels) == 30000.0
        assert reg.get_gauge('hg_strategy_last_signal_timestamp', labels) == 1713400000.0

    def test_update_position(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_position('AAPL', 100.0, 250.0)
        labels = {'symbol': 'AAPL', 'strategy': 'omr'}
        assert reg.get_gauge('hg_position_qty', labels) == 100.0
        assert reg.get_gauge('hg_position_unrealized_pnl_usd', labels) == 250.0

    def test_close_position_removes_metrics(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_position('AAPL', 100.0, 250.0)
        reg.close_position('AAPL')
        labels = {'symbol': 'AAPL', 'strategy': 'omr'}
        assert reg.get_gauge('hg_position_qty', labels) is None
        assert reg.get_gauge('hg_position_unrealized_pnl_usd', labels) is None

    def test_record_order_submitted(self):
        reg = MetricsRegistry(strategy='omr')
        reg.record_order_submitted('buy', 'alpaca')
        assert reg.get_counter(
            'hg_orders_submitted_total',
            {'strategy': 'omr', 'side': 'buy', 'broker': 'alpaca'}
        ) == 1

    def test_record_order_filled(self):
        reg = MetricsRegistry(strategy='omr')
        reg.record_order_filled('alpaca', 3.5)
        assert reg.get_counter(
            'hg_orders_filled_total',
            {'strategy': 'omr', 'broker': 'alpaca'}
        ) == 1
        hist = reg.get_histogram('hg_fill_slippage_bps', {'strategy': 'omr'})
        assert hist['count'] == 1
        assert hist['sum'] == 3.5

    def test_record_order_rejected(self):
        reg = MetricsRegistry(strategy='omr')
        reg.record_order_rejected('insufficient_funds', 'alpaca')
        assert reg.get_counter(
            'hg_orders_rejected_total',
            {'strategy': 'omr', 'reason': 'insufficient_funds', 'broker': 'alpaca'}
        ) == 1

    def test_update_regime(self):
        reg = MetricsRegistry(strategy='ramp')
        reg.update_regime(state_code=0, sma_20=1.02, sma_50=1.01,
                          sma_200=0.98, time_in_state_seconds=3600.0)
        assert reg.get_gauge('hg_regime_state_code') == 0
        assert reg.get_gauge('hg_regime_sma_signal', {'period': '20'}) == 1.02
        assert reg.get_gauge('hg_regime_time_in_state_seconds') == 3600.0

    def test_update_websocket(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_websocket('iex', connected=True, symbols=500)
        assert reg.get_gauge('hg_websocket_connected', {'provider': 'iex'}) == 1.0
        assert reg.get_gauge('hg_websocket_symbols_subscribed', {'provider': 'iex'}) == 500

    def test_update_market_open(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_market_open(True)
        assert reg.get_gauge('hg_market_open') == 1.0
        reg.update_market_open(False)
        assert reg.get_gauge('hg_market_open') == 0.0

    def test_update_broker_heartbeat(self):
        import time
        reg = MetricsRegistry(strategy='omr')
        reg.update_broker_heartbeat('alpaca')
        ts = reg.get_gauge('hg_broker_last_heartbeat_timestamp', {'broker': 'alpaca'})
        assert ts is not None
        # Gauge stores Unix epoch; age (time() - ts) must be non-negative and small
        assert 0.0 <= (time.time() - ts) < 5.0

    def test_update_process_rss(self):
        reg = MetricsRegistry(strategy='omr')
        reg.update_process_rss()
        rss = reg.get_gauge('hg_process_rss_bytes', {'strategy': 'omr'})
        assert rss is not None
        assert rss > 0

    def test_update_ramp_cache(self):
        reg = MetricsRegistry(strategy='ramp')
        reg.update_ramp_cache(age_seconds=120.0, hit=True)
        assert reg.get_gauge('hg_ramp_cache_age_seconds') == 120.0
        assert reg.get_counter('hg_ramp_cache_hit_total') == 1
