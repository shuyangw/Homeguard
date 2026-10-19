"""CSCM trailing stop must not fire when held positions cannot be priced.

On 2026-10-16 the instance lost DNS, DemoBroker returned last=0.0 quotes, the
positions were valued at $0, and the stop fired on an 88.8% "drawdown" that was
really just the cash balance.
"""
from decimal import Decimal

import pytest

from src.trading.brokers.interfaces.base import OrderSide
from src.trading.demo.demo_broker import DemoBroker
from tests.trading.test_cscm_position_sizing import PRICES, UNIVERSE, _make_adapter, _seed_broker_prices


@pytest.fixture
def adapter_holding_positions(tmp_path, monkeypatch):
    from src.trading.demo.portfolio_state import DemoPortfolioState
    from src.trading.demo.trade_logger import DemoTradeLogger
    monkeypatch.setattr(DemoPortfolioState, 'DEFAULT_STATE_DIR', tmp_path)
    monkeypatch.setattr(DemoTradeLogger, 'DEFAULT_LOG_DIR', tmp_path / 'trades')

    broker = DemoBroker(initial_cash=100000.0, slippage_bps=0.0, fee_bps=0.0)
    _seed_broker_prices(broker, PRICES)
    for symbol in UNIVERSE:
        quantity = Decimal(str(round(10000.0 / PRICES[symbol], 6)))
        broker.place_crypto_order(symbol, quantity, OrderSide.BUY)
    adapter = _make_adapter(broker)
    assert adapter._check_trailing_stop() is False  # records the peak
    return adapter, broker


def _quotes_at(broker, prices):
    real_quote = broker.get_crypto_quote

    def quote(symbol):
        q = dict(real_quote(symbol))
        q['last'] = prices.get(symbol, 0.0)
        return q
    return quote


def test_unpriced_position_does_not_trigger_trailing_stop(adapter_holding_positions, monkeypatch):
    adapter, broker = adapter_holding_positions
    outage = {}  # no symbol can be priced, as during the DNS outage
    monkeypatch.setattr(broker, 'get_crypto_quote', _quotes_at(broker, outage))

    assert adapter._check_trailing_stop() is False
    assert not adapter.signals.is_trailing_stop_active()
    assert len(broker.get_crypto_positions()) == len(UNIVERSE)


def test_real_drawdown_still_triggers_trailing_stop(adapter_holding_positions, monkeypatch):
    adapter, broker = adapter_holding_positions
    crashed = {s: p * 0.5 for s, p in PRICES.items()}
    monkeypatch.setattr(broker, 'get_crypto_quote', _quotes_at(broker, crashed))

    assert adapter._check_trailing_stop() is True
    assert adapter.signals.is_trailing_stop_active()
