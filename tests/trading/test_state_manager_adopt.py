"""Tests for adopt_broker_positions: backfilling state from broker holdings.

Closes the desync where a strategy holds broker positions opened in a prior
process (or held across a restart) that were never written into the state
`positions` dict. sync_with_broker only reconciles already-tracked symbols, so
it cannot heal a state whose positions dict is empty while the broker holds
shares. adopt_broker_positions adds those untracked-but-held symbols so that
get_positions() -- and therefore strategy-equity and ownership accounting --
reflect reality.
"""

import tempfile
from pathlib import Path

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


def test_adopt_adds_untracked_broker_position(mgr):
    # State is empty for ramp; broker reports a held position.
    broker_positions = {'AMZN': {'quantity': 12, 'avg_entry_price': 185.50}}
    adopted = mgr.adopt_broker_positions('ramp', 'ibkr', broker_positions)

    assert adopted == ['AMZN']
    pos = mgr.get_positions('ramp')['AMZN']
    assert pos['qty'] == 12
    assert pos['entry_price'] == 185.50
    assert pos['broker'] == 'ibkr'


def test_adopt_skips_symbol_already_tracked_by_strategy(mgr):
    mgr.add_position('ramp', 'AMZN', 12, 185.50, order_id='x', broker='ibkr')
    # Broker reports the same symbol; adoption must NOT duplicate or alter qty.
    broker_positions = {'AMZN': {'quantity': 12, 'avg_entry_price': 999.0}}
    adopted = mgr.adopt_broker_positions('ramp', 'ibkr', broker_positions)

    assert adopted == []
    assert mgr.get_positions('ramp')['AMZN']['qty'] == 12
    assert mgr.get_positions('ramp')['AMZN']['entry_price'] == 185.50


def test_adopt_skips_symbol_owned_by_other_strategy(mgr):
    mgr.add_position('omr', 'TQQQ', 100, 52.30, order_id='x', broker='ibkr')
    broker_positions = {'TQQQ': {'quantity': 100, 'avg_entry_price': 52.30}}
    adopted = mgr.adopt_broker_positions('ramp', 'ibkr', broker_positions)

    assert adopted == []
    assert 'TQQQ' not in mgr.get_positions('ramp')
    assert mgr.symbol_owned_by_other('ramp', 'TQQQ') == 'omr'


def test_adopt_skips_zero_quantity(mgr):
    broker_positions = {'AMZN': {'quantity': 0, 'avg_entry_price': 185.50}}
    adopted = mgr.adopt_broker_positions('ramp', 'ibkr', broker_positions)

    assert adopted == []
    assert 'AMZN' not in mgr.get_positions('ramp')


def test_adopted_position_is_reconcilable_by_sync(mgr):
    # After adoption, a later broker close must be reconciled by sync_with_broker
    # (proves the adopted row is a first-class tracked position, broker-tagged).
    mgr.adopt_broker_positions(
        'ramp', 'ibkr', {'AMZN': {'quantity': 12, 'avg_entry_price': 185.50}}
    )
    changes = mgr.sync_with_broker('ibkr', {})  # broker now flat
    assert 'ramp:AMZN' in changes['removed']
    assert 'AMZN' not in mgr.get_positions('ramp')


def test_adopt_multiple_positions_returns_all(mgr):
    broker_positions = {
        'AMZN': {'quantity': 12, 'avg_entry_price': 185.50},
        'META': {'quantity': 5, 'avg_entry_price': 620.00},
    }
    adopted = mgr.adopt_broker_positions('ramp', 'ibkr', broker_positions)

    assert sorted(adopted) == ['AMZN', 'META']
    assert set(mgr.get_positions('ramp').keys()) == {'AMZN', 'META'}
