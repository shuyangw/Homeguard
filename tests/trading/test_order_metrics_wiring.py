"""Tests for order-event metric wiring and the bounded reject-reason label.

Two defects are guarded here:

1. ExecutionEngine has always accepted `metrics_registry=`, but StrategyAdapter
   constructed it as `ExecutionEngine(broker)`, so no live process ever recorded
   hg_orders_submitted_total / hg_orders_filled_total / hg_orders_rejected_total /
   hg_fill_slippage_bps. The metrics existed in the registry and were dead in
   production.

2. The rejection `reason` label was `str(e)[:50]`, i.e. raw broker error text.
   Broker errors embed order IDs and symbols, so that is unbounded cardinality
   on a counter in a 90-day-retention TSDB. An exploding label set is worse than
   a missing metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.brokers.broker_interface import (
    BrokerError,
    InvalidOrderError,
    OrderSide,
)
from src.trading.core.execution_engine import (
    ExecutionEngine,
    _REJECT_REASON_PATTERNS,
    classify_reject_reason,
)

VALID_REASONS = {label for label, _ in _REJECT_REASON_PATTERNS} | {'other'}


class SpyRegistry:
    """Records the calls ExecutionEngine makes, mimicking MetricsRegistry."""

    def __init__(self):
        self.submitted: list[tuple] = []
        self.filled: list[tuple] = []
        self.rejected: list[tuple] = []

    def record_order_submitted(self, side, broker):
        self.submitted.append((side, broker))

    def record_order_filled(self, slippage_bps, broker):
        self.filled.append((slippage_bps, broker))

    def record_order_rejected(self, reason, broker):
        self.rejected.append((reason, broker))


class TestRejectReasonClassifier:
    """The label must come from a closed set, whatever the broker says."""

    @pytest.mark.parametrize('message,expected', [
        ('Insufficient buying power for order', 'insufficient_funds'),
        ('insufficient funds', 'insufficient_funds'),
        ('Invalid quantity: 0', 'invalid_quantity'),
        ('limit price required for LIMIT order', 'missing_price'),
        ('No security definition has been found', 'not_tradable'),
        ('Symbol is halted', 'not_tradable'),
        ('Rate limit exceeded, retry later', 'rate_limited'),
        ('pacing violation', 'rate_limited'),
        ('Connection reset by peer', 'connection'),
        ('request timed out after 30s', 'connection'),
        ('Order was rejected by exchange', 'rejected'),
        ('something nobody has ever seen', 'other'),
    ])
    def test_known_messages_map_to_stable_labels(self, message, expected):
        assert classify_reject_reason(BrokerError(message)) == expected

    def test_every_result_is_in_the_closed_set(self):
        """The whole point: no input can produce a novel label."""
        adversarial = [
            'Order 1234567890 for AAPL rejected at 2026-07-27T15:55:01.123Z',
            'ERR-99: unmapped condition xyz',
            '',
            'a' * 5000,
            'unicode: \xe9\xe8\xea',  # escaped so the source file stays ASCII
        ]
        for message in adversarial:
            assert classify_reject_reason(BrokerError(message)) in VALID_REASONS

    def test_order_ids_do_not_leak_into_the_label(self):
        """This is the cardinality bug: the old code used str(e)[:50] directly."""
        reason = classify_reject_reason(
            BrokerError('Order 987654321 for TSLA rejected: insufficient buying power'))
        assert reason == 'insufficient_funds'
        assert '987654321' not in reason
        assert 'TSLA' not in reason

    def test_classifier_never_raises_on_odd_exceptions(self):
        class Weird(Exception):
            def __str__(self):
                return 'connection lost'

        assert classify_reject_reason(Weird()) == 'connection'


class TestExecutionEngineRecordsRejections:
    def test_invalid_order_records_bounded_reason(self):
        """InvalidOrderError takes the no-retry path and must still record."""
        registry = SpyRegistry()

        class RejectingBroker:
            name = 'testbroker'

            def place_stock_order(self, **kwargs):
                raise InvalidOrderError('Insufficient buying power for 100 AAPL')

        engine = ExecutionEngine(RejectingBroker(), max_retries=1, retry_delay=0,
                                 metrics_registry=registry)
        with pytest.raises(InvalidOrderError):
            engine.execute_order('AAPL', 100, OrderSide.BUY)

        assert len(registry.rejected) == 1
        reason, broker = registry.rejected[0]
        assert reason == 'insufficient_funds'
        assert reason in VALID_REASONS
        assert broker == 'testbroker'

    def test_no_registry_means_no_crash(self):
        """metrics_registry defaults to None; that path must stay silent."""

        class RejectingBroker:
            name = 'testbroker'

            def place_stock_order(self, **kwargs):
                raise InvalidOrderError('Invalid quantity: 0')

        engine = ExecutionEngine(RejectingBroker(), max_retries=1, retry_delay=0)
        assert engine.metrics_registry is None
        with pytest.raises(InvalidOrderError):
            engine.execute_order('AAPL', 0, OrderSide.BUY)

    def test_metrics_failure_does_not_break_execution(self):
        """A broken registry must not turn a rejection into a different error."""

        class ExplodingRegistry:
            def record_order_submitted(self, side, broker):
                raise RuntimeError('metrics backend down')

            def record_order_rejected(self, reason, broker):
                raise RuntimeError('metrics backend down')

        class RejectingBroker:
            name = 'testbroker'

            def place_stock_order(self, **kwargs):
                raise InvalidOrderError('halted')

        engine = ExecutionEngine(RejectingBroker(), max_retries=1, retry_delay=0,
                                 metrics_registry=ExplodingRegistry())
        # The original InvalidOrderError must surface, not the metrics RuntimeError.
        with pytest.raises(InvalidOrderError):
            engine.execute_order('AAPL', 1, OrderSide.BUY)


class TestStrategyAdapterForwardsRegistry:
    """The actual regression guard for the dead-metrics defect."""

    def _adapter_class(self):
        from src.trading.adapters.strategy_adapter import StrategyAdapter

        class Bare(StrategyAdapter):
            def get_signals(self, *a, **k):
                return []

            def run_once(self, *a, **k):
                return {}

            def get_schedule(self):
                return {}

        return Bare

    def _stub_args(self):
        from tests.trading.mock_broker import MockBroker

        class Strategy:
            def generate_signals(self, *a, **k):
                return []

        return Strategy(), MockBroker(initial_cash=100000.0), ['AAPL']

    def test_registry_reaches_execution_engine(self):
        sentinel = SpyRegistry()
        strategy, broker, symbols = self._stub_args()
        adapter = self._adapter_class()(
            strategy=strategy, broker=broker, symbols=symbols,
            metrics_registry=sentinel)
        assert adapter.execution_engine.metrics_registry is sentinel, (
            'StrategyAdapter must forward metrics_registry to ExecutionEngine; '
            'without this every hg_orders_* metric is dead in production'
        )
        assert adapter._metrics_registry is sentinel

    def test_omitting_registry_preserves_previous_behaviour(self):
        strategy, broker, symbols = self._stub_args()
        adapter = self._adapter_class()(
            strategy=strategy, broker=broker, symbols=symbols)
        assert adapter.execution_engine.metrics_registry is None
        assert adapter._metrics_registry is None
