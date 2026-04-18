"""Tests for metrics HTTP server."""

import threading
import time
import urllib.request
import json

from src.monitoring.registry import MetricsRegistry
from src.monitoring.server import start_metrics_server


class TestMetricsServer:
    """Test HTTP endpoint contracts."""

    # Incremented per setup so each test binds a unique port; on Windows
    # HTTPServer does not set SO_EXCLUSIVEADDRUSE, so reusing one port
    # across serial tests causes unpredictable routing between
    # still-running daemon servers from prior tests.
    _next_port = 18081

    def setup_method(self):
        """Start server on a unique port for each test."""
        self.registry = MetricsRegistry(strategy='omr')
        self.port = TestMetricsServer._next_port
        TestMetricsServer._next_port += 1
        self.thread = start_metrics_server(
            self.registry, host='127.0.0.1', port=self.port
        )
        # Give server time to start
        time.sleep(0.3)

    def teardown_method(self):
        """Server thread is daemon, will die with test process."""
        pass

    def test_health_endpoint(self):
        url = f'http://127.0.0.1:{self.port}/health'
        with urllib.request.urlopen(url, timeout=2) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data['status'] == 'ok'
            assert data['strategy'] == 'omr'

    def test_metrics_endpoint_empty(self):
        url = f'http://127.0.0.1:{self.port}/metrics'
        with urllib.request.urlopen(url, timeout=2) as resp:
            assert resp.status == 200
            content_type = resp.headers.get('Content-Type', '')
            assert 'text/plain' in content_type

    def test_metrics_endpoint_with_data(self):
        self.registry.set_gauge('hg_portfolio_equity_usd', 100000.0,
                                {'broker': 'alpaca'})
        self.registry.inc_counter('hg_orders_filled_total',
                                  {'strategy': 'omr', 'broker': 'alpaca'})

        url = f'http://127.0.0.1:{self.port}/metrics'
        with urllib.request.urlopen(url, timeout=2) as resp:
            body = resp.read().decode('utf-8')
            assert 'hg_portfolio_equity_usd{broker="alpaca"} 100000.0' in body
            assert 'hg_orders_filled_total{broker="alpaca",strategy="omr"} 1' in body

    def test_unknown_path_returns_404(self):
        url = f'http://127.0.0.1:{self.port}/unknown'
        try:
            urllib.request.urlopen(url, timeout=2)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_server_thread_is_daemon(self):
        assert self.thread.daemon is True
