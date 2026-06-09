"""Integration tests for runner pre-flight reconciliation."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.trading.state.strategy_state_manager import StrategyStateManager
from scripts.trading.run_live_paper_trading import preflight_reconcile


@pytest.fixture
def mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_file = tmp / "strategy_positions.json"
        toggle_file = tmp / "strategy_toggle.yaml"
        toggle_file.write_text(
            "strategies:\n  omr:\n    enabled: true\n"
        )
        yield StrategyStateManager(state_file=state_file, toggle_file=toggle_file)


def _fake_broker(positions):
    b = Mock()
    b.get_stock_positions.return_value = [
        {'symbol': s, 'quantity': q} for s, q in positions.items()
    ]
    return b


def test_preflight_passes_when_flat(mgr):
    broker = _fake_broker({})
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False)
    assert rc == 0


def test_preflight_passes_when_state_matches_broker(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    broker = _fake_broker({'TQQQ': 100})
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False)
    assert rc == 0


def test_preflight_blocks_cross_broker_mismatch(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    ibkr = _fake_broker({})
    rc = preflight_reconcile('omr', ibkr, 'ibkr', mgr, force_start=False)
    assert rc == 1


def test_force_start_bypasses_mismatch_without_mutating_state(mgr, tmp_path):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    state_before = mgr.state_file.read_text()

    ibkr = _fake_broker({})
    rc = preflight_reconcile('omr', ibkr, 'ibkr', mgr, force_start=True)

    state_after = mgr.state_file.read_text()
    assert rc == 0
    assert state_before == state_after, "Pre-flight must not mutate state"


def test_preflight_blocks_on_state_broker_qty_zero(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    broker = _fake_broker({})
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False, retry_delay=0)
    assert rc == 1


def test_preflight_retries_transient_empty_then_passes(mgr):
    """A freshly-connected IBKR session can report 0 positions until the async
    account download (accountDownloadEnd) completes. Preflight must retry the
    broker query before trusting a 0, or a cold-start race trips a false
    mismatch and crash-loops the runner.
    """
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='ibkr')
    broker = Mock()
    broker.get_stock_positions.side_effect = [
        [],  # attempt 1: portfolio() not populated yet
        [],  # attempt 2: still warming
        [{'symbol': 'TQQQ', 'quantity': 100}],  # attempt 3: populated
    ]
    rc = preflight_reconcile('omr', broker, 'ibkr', mgr, force_start=False, retry_delay=0)
    assert rc == 0
    assert broker.get_stock_positions.call_count == 3


def test_preflight_blocks_after_retries_when_persistently_empty(mgr):
    """If the broker is genuinely flat after all retries, the real mismatch
    still blocks (the retry must not mask a true position loss)."""
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='ibkr')
    broker = _fake_broker({})  # always empty
    rc = preflight_reconcile('omr', broker, 'ibkr', mgr, force_start=False, retry_delay=0)
    assert rc == 1
    assert broker.get_stock_positions.call_count >= 2  # retried before blocking


def test_preflight_no_retry_when_positions_on_other_broker(mgr):
    """An empty result from THIS broker is expected when state positions are on
    another broker; no retry -- block immediately on the cross-broker tag."""
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    broker = Mock()
    broker.get_stock_positions.return_value = []
    rc = preflight_reconcile('omr', broker, 'ibkr', mgr, force_start=False, retry_delay=0)
    assert rc == 1
    assert broker.get_stock_positions.call_count == 1  # no retry for cross-broker


def test_broker_unreachable_blocks_without_force(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    broker = Mock()
    broker.get_stock_positions.side_effect = RuntimeError("connection refused")
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False)
    assert rc == 1


def test_broker_unreachable_with_force_proceeds(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    broker = Mock()
    broker.get_stock_positions.side_effect = RuntimeError("connection refused")
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=True)
    assert rc == 0


def test_force_start_mismatch_does_not_log_success(mgr):
    from unittest.mock import patch
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    ibkr = _fake_broker({})
    with patch('scripts.trading.run_live_paper_trading.logger') as mock_logger:
        rc = preflight_reconcile('omr', ibkr, 'ibkr', mgr, force_start=True)
    assert rc == 0
    assert not any(
        'Pre-flight check passed' in str(call)
        for call in mock_logger.success.call_args_list
    )


def test_preflight_handles_missing_broker_tag_gracefully(mgr):
    # Seed a state entry with no 'broker' key (legacy/hand-edited)
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    mgr._load_state()
    del mgr._state['strategies']['omr']['positions']['TQQQ']['broker']
    mgr._save_state()

    broker = _fake_broker({'TQQQ': 100})
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False)
    assert rc == 1  # missing broker tag must block


def test_preflight_handles_missing_broker_tag_with_force(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    mgr._load_state()
    del mgr._state['strategies']['omr']['positions']['TQQQ']['broker']
    mgr._save_state()

    broker = _fake_broker({'TQQQ': 100})
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=True)
    assert rc == 0
