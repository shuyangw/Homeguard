"""Tests for BrokerRoutingMap broker-name resolution."""

from unittest.mock import Mock

from src.trading.config.broker_routing import BrokerRoutingMap


def test_getitem_returns_instance():
    alpaca = Mock(name='alpaca-instance')
    rmap = BrokerRoutingMap({'omr': ('alpaca', alpaca)}, ('alpaca', alpaca))
    assert rmap['omr'] is alpaca


def test_get_broker_name_returns_name():
    alpaca = Mock(name='alpaca-instance')
    ibkr = Mock(name='ibkr-instance')
    rmap = BrokerRoutingMap(
        {'omr': ('alpaca', alpaca), 'ramp': ('ibkr', ibkr)},
        ('alpaca', alpaca),
    )
    assert rmap.get_broker_name('omr') == 'alpaca'
    assert rmap.get_broker_name('ramp') == 'ibkr'


def test_get_broker_name_falls_back_to_default():
    alpaca = Mock(name='alpaca-instance')
    rmap = BrokerRoutingMap({}, ('alpaca', alpaca))
    assert rmap.get_broker_name('unknown_strategy') == 'alpaca'
    assert rmap['unknown_strategy'] is alpaca


def test_contains():
    alpaca = Mock()
    rmap = BrokerRoutingMap({'omr': ('alpaca', alpaca)}, ('alpaca', alpaca))
    assert 'omr' in rmap
    assert 'ramp' not in rmap
