"""Tests for broker-aware sync_with_broker."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.trading.state.strategy_state_manager import StrategyStateManager


@pytest.fixture
def mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        state_file = tmp / "strategy_positions.json"
        toggle_file = tmp / "strategy_toggle.yaml"
        toggle_file.write_text(
            "strategies:\n"
            "  omr:\n    enabled: true\n"
            "  mp:\n    enabled: true\n"
            "  ramp:\n    enabled: true\n"
        )
        yield StrategyStateManager(state_file=state_file, toggle_file=toggle_file)


def test_sync_removes_closed_position_on_matching_broker(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    changes = mgr.sync_with_broker('alpaca', {})
    assert 'omr:TQQQ' in changes['removed']
    assert 'TQQQ' not in mgr.get_positions('omr')


def test_sync_skips_positions_on_different_broker(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    # Runner is on ibkr; ibkr reports empty.
    changes = mgr.sync_with_broker('ibkr', {})
    assert changes['removed'] == []
    assert 'omr:TQQQ (on alpaca)' in changes['skipped']
    # Position must still be there.
    assert mgr.get_positions('omr')['TQQQ']['qty'] == 100


def test_sync_updates_reduced_qty(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    changes = mgr.sync_with_broker('alpaca', {'TQQQ': 40})
    assert 'omr:TQQQ' in changes['updated']
    assert mgr.get_positions('omr')['TQQQ']['qty'] == 40


def test_sync_detects_drift_on_increase(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    changes = mgr.sync_with_broker('alpaca', {'TQQQ': 150})
    assert 'omr:TQQQ' in changes['drift_detected']
    assert mgr.get_positions('omr')['TQQQ']['qty'] == 150


def test_sync_raises_on_untagged_position(mgr):
    # Simulate a post-migration bug: inject an untagged position directly.
    mgr._load_state()
    mgr._state['strategies']['omr'] = {
        'positions': {
            'TQQQ': {
                'qty': 100,
                'entry_price': 52.30,
                'entry_time': '2026-04-14T15:50:00-04:00',
                'order_id': 'x',
                # no broker field
            }
        },
        'last_execution': None,
    }
    mgr._save_state()

    with pytest.raises(ValueError, match="Untagged position"):
        mgr.sync_with_broker('alpaca', {'TQQQ': 100})


def _fake_broker(positions):
    b = Mock()
    b.get_stock_positions.return_value = [
        {'symbol': s, 'quantity': q} for s, q in positions.items()
    ]
    return b


def test_get_positions_by_broker_groups(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='a', broker='alpaca')
    mgr.add_position('omr', 'SPY', 50, 500.00, order_id='b', broker='ibkr')
    grouped = mgr.get_positions_by_broker('omr')
    assert set(grouped.keys()) == {'alpaca', 'ibkr'}
    assert 'TQQQ' in grouped['alpaca']
    assert 'SPY' in grouped['ibkr']


def test_check_broker_switch_safety_safe_when_flat(mgr):
    current = _fake_broker({}); new = _fake_broker({})
    result = mgr.check_broker_switch_safety('omr', 'alpaca', 'ibkr', current, new)
    assert result['safe'] is True
    assert result['blocking_reasons'] == []


def test_check_broker_switch_safety_blocks_when_current_holds(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    current = _fake_broker({'TQQQ': 100}); new = _fake_broker({})
    result = mgr.check_broker_switch_safety('omr', 'alpaca', 'ibkr', current, new)
    assert result['safe'] is False
    assert any('TQQQ' in r and '100' in r for r in result['blocking_reasons'])


def test_check_broker_switch_safety_unreachable_current_blocks(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='alpaca')
    current = Mock()
    current.get_stock_positions.side_effect = RuntimeError("connection refused")
    new = _fake_broker({})
    result = mgr.check_broker_switch_safety('omr', 'alpaca', 'ibkr', current, new)
    assert result['safe'] is False
    assert any('Cannot reach' in r for r in result['blocking_reasons'])
