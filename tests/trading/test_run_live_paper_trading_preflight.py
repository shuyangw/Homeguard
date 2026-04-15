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
    rc = preflight_reconcile('omr', broker, 'alpaca', mgr, force_start=False)
    assert rc == 1


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
