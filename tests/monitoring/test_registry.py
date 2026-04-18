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
