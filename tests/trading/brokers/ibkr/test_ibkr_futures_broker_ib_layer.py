"""Unit tests for IBKRFuturesBroker's ib_async-layer helpers.

These tests inject a mock IBKRConnectionManager so the broker thinks it
has a live IB connection but the underlying calls hit MagicMock. They
verify translation, filtering, order construction, and how the broker
maps abstract enums to ib_async types -- without ever talking to IBKR.
"""
from unittest.mock import MagicMock

import pytest

from src.trading.brokers.ibkr.config import IBKRConfig
from src.trading.brokers.ibkr.ibkr_futures_broker import (
    IBKRFuturesBroker, _EXCHANGE_BY_ROOT, _DEFAULT_EXCHANGE,
)
from src.trading.brokers.interfaces.base import OrderSide, OrderType, TimeInForce


def _mk_broker_with_mock_conn():
    """Broker with mocked _conn so no real IBKR connection is opened."""
    broker = IBKRFuturesBroker(config=IBKRConfig(port=4002))
    mock_conn = MagicMock()
    broker._conn = mock_conn
    return broker, mock_conn


# --- exchange routing -----------------------------------------------------

def test_exchange_routing_cme():
    assert IBKRFuturesBroker._exchange_for("ES") == "CME"
    assert IBKRFuturesBroker._exchange_for("MES") == "CME"
    assert IBKRFuturesBroker._exchange_for("6E") == "CME"


def test_exchange_routing_nymex():
    assert IBKRFuturesBroker._exchange_for("CL") == "NYMEX"
    assert IBKRFuturesBroker._exchange_for("NG") == "NYMEX"


def test_exchange_routing_comex():
    assert IBKRFuturesBroker._exchange_for("GC") == "COMEX"
    assert IBKRFuturesBroker._exchange_for("SI") == "COMEX"


def test_exchange_routing_cbot():
    assert IBKRFuturesBroker._exchange_for("ZN") == "CBOT"
    assert IBKRFuturesBroker._exchange_for("ZC") == "CBOT"
    assert IBKRFuturesBroker._exchange_for("10Y") == "CBOT"


def test_exchange_routing_unknown_falls_back():
    assert IBKRFuturesBroker._exchange_for("UNKNOWN_ROOT") == _DEFAULT_EXCHANGE


# --- side / tif translation -----------------------------------------------

def test_side_to_ibkr():
    assert IBKRFuturesBroker._side_to_ibkr(OrderSide.BUY) == "BUY"
    assert IBKRFuturesBroker._side_to_ibkr(OrderSide.SELL) == "SELL"


def test_tif_to_ibkr():
    assert IBKRFuturesBroker._tif_to_ibkr(TimeInForce.DAY) == "DAY"
    assert IBKRFuturesBroker._tif_to_ibkr(TimeInForce.GTC) == "GTC"
    assert IBKRFuturesBroker._tif_to_ibkr(TimeInForce.IOC) == "IOC"
    assert IBKRFuturesBroker._tif_to_ibkr(TimeInForce.FOK) == "FOK"


# --- _build_order ---------------------------------------------------------

def test_build_order_limit_requires_price():
    broker, _ = _mk_broker_with_mock_conn()
    with pytest.raises(ValueError, match="Limit price required"):
        broker._build_order("BUY", 1, OrderType.LIMIT, None, None, TimeInForce.DAY)


def test_build_order_stop_requires_price():
    broker, _ = _mk_broker_with_mock_conn()
    with pytest.raises(ValueError, match="Stop price required"):
        broker._build_order("BUY", 1, OrderType.STOP, None, None, TimeInForce.DAY)


def test_build_order_stop_limit_requires_both():
    broker, _ = _mk_broker_with_mock_conn()
    with pytest.raises(ValueError, match="Both prices required"):
        broker._build_order("BUY", 1, OrderType.STOP_LIMIT, None, 100.0, TimeInForce.DAY)


def test_build_order_market_no_price_needed():
    broker, _ = _mk_broker_with_mock_conn()
    o = broker._build_order("BUY", 1, OrderType.MARKET, None, None, TimeInForce.DAY)
    assert o.action == "BUY"
    assert o.totalQuantity == 1
    assert o.tif == "DAY"


def test_build_order_what_if_flag():
    broker, _ = _mk_broker_with_mock_conn()
    o = broker._build_order(
        "BUY", 1, OrderType.MARKET, None, None, TimeInForce.DAY, what_if=True,
    )
    assert o.whatIf is True


# --- _translate_trade -----------------------------------------------------

def test_translate_trade_basic():
    broker, _ = _mk_broker_with_mock_conn()
    trade = MagicMock()
    trade.order.orderId = 42
    trade.order.permId = 99
    trade.order.totalQuantity = 3
    trade.order.action = "BUY"
    trade.order.orderType = "LMT"
    trade.order.lmtPrice = 5300.0
    trade.order.auxPrice = 0
    trade.orderStatus.status = "Submitted"
    trade.orderStatus.filled = 0
    trade.orderStatus.avgFillPrice = 0
    trade.contract.localSymbol = "ESM4"
    trade.contract.symbol = "ES"
    trade.contract.lastTradeDateOrContractMonth = "202406"
    d = broker._translate_trade(trade)
    assert d["orderId"] == 42
    assert d["status"] == "pending"
    assert d["raw_status"] == "Submitted"
    assert d["symbol"] == "ESM4"
    assert d["contract_month"] == "202406"
    assert d["limit_price"] == 5300.0


def test_translate_trade_filled():
    broker, _ = _mk_broker_with_mock_conn()
    trade = MagicMock()
    trade.order.orderId = 1
    trade.order.permId = 2
    trade.order.totalQuantity = 1
    trade.order.action = "SELL"
    trade.order.orderType = "MKT"
    trade.order.lmtPrice = 0
    trade.order.auxPrice = 0
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 1
    trade.orderStatus.avgFillPrice = 5301.5
    trade.contract.localSymbol = "ESM4"
    trade.contract.symbol = "ES"
    trade.contract.lastTradeDateOrContractMonth = "202406"
    d = broker._translate_trade(trade)
    assert d["status"] == "filled"
    assert d["filled_qty"] == 1
    assert d["filled_avg_price"] == 5301.5
    assert d["side"] == "sell"


# --- cancel_order / get_order / get_orders --------------------------------

def test_cancel_order_finds_and_cancels():
    broker, mock_conn = _mk_broker_with_mock_conn()
    trade = MagicMock()
    trade.order.orderId = 42
    mock_conn.ib.openTrades.return_value = [trade]
    assert broker.cancel_order("42") is True
    mock_conn.ib.cancelOrder.assert_called_once_with(trade.order)


def test_cancel_order_not_found_returns_false():
    broker, mock_conn = _mk_broker_with_mock_conn()
    other = MagicMock()
    other.order.orderId = 999
    mock_conn.ib.openTrades.return_value = [other]
    assert broker.cancel_order("42") is False


def test_get_order_returns_translated():
    broker, mock_conn = _mk_broker_with_mock_conn()
    trade = MagicMock()
    trade.order.orderId = 42
    trade.order.permId = 43
    trade.order.totalQuantity = 1
    trade.order.action = "BUY"
    trade.order.orderType = "MKT"
    trade.order.lmtPrice = 0
    trade.order.auxPrice = 0
    trade.orderStatus.status = "Submitted"
    trade.orderStatus.filled = 0
    trade.orderStatus.avgFillPrice = 0
    trade.contract.localSymbol = "ESM4"
    trade.contract.symbol = "ES"
    trade.contract.lastTradeDateOrContractMonth = "202406"
    mock_conn.ib.trades.return_value = [trade]
    d = broker.get_order("42")
    assert d["orderId"] == 42


def test_get_order_missing_raises():
    broker, mock_conn = _mk_broker_with_mock_conn()
    mock_conn.ib.trades.return_value = []
    with pytest.raises(LookupError, match="Order 42 not found"):
        broker.get_order("42")


def test_get_orders_filters_to_futures():
    """Stock trades returned by IB should be excluded from get_orders."""
    broker, mock_conn = _mk_broker_with_mock_conn()
    fut_trade = MagicMock()
    fut_trade.order.orderId = 1
    fut_trade.order.permId = 1
    fut_trade.order.totalQuantity = 1
    fut_trade.order.action = "BUY"
    fut_trade.order.orderType = "MKT"
    fut_trade.order.lmtPrice = 0
    fut_trade.order.auxPrice = 0
    fut_trade.orderStatus.status = "Submitted"
    fut_trade.orderStatus.filled = 0
    fut_trade.orderStatus.avgFillPrice = 0
    fut_trade.contract.secType = "FUT"
    fut_trade.contract.localSymbol = "ESM4"
    fut_trade.contract.symbol = "ES"
    fut_trade.contract.lastTradeDateOrContractMonth = "202406"

    stk_trade = MagicMock()
    stk_trade.contract.secType = "STK"

    mock_conn.ib.trades.return_value = [fut_trade, stk_trade]
    orders = broker.get_orders()
    assert len(orders) == 1
    assert orders[0]["symbol"] == "ESM4"


# --- positions ------------------------------------------------------------

def test_get_futures_positions_filters_to_futures():
    broker, mock_conn = _mk_broker_with_mock_conn()
    fut_pos = MagicMock()
    fut_pos.position = 2
    fut_pos.averageCost = 5300.0
    fut_pos.marketPrice = 5310.0
    fut_pos.marketValue = 10620.0
    fut_pos.unrealizedPNL = 20.0
    fut_pos.contract.secType = "FUT"
    fut_pos.contract.symbol = "ES"
    fut_pos.contract.lastTradeDateOrContractMonth = "202406"
    fut_pos.contract.localSymbol = "ESM4"
    fut_pos.contract.multiplier = 50

    stk_pos = MagicMock()
    stk_pos.contract.secType = "STK"

    mock_conn.ib.portfolio.return_value = [fut_pos, stk_pos]
    out = broker.get_futures_positions()
    assert len(out) == 1
    assert out[0]["symbol_root"] == "ES"
    assert out[0]["contract_month"] == "202406"
    assert out[0]["quantity"] == 2
    assert out[0]["avg_entry_price"] == 5300.0


def test_get_futures_position_lookup():
    broker, mock_conn = _mk_broker_with_mock_conn()
    fut_pos = MagicMock()
    fut_pos.position = 1
    fut_pos.averageCost = 5300.0
    fut_pos.marketPrice = 5310.0
    fut_pos.marketValue = 5310.0
    fut_pos.unrealizedPNL = 10.0
    fut_pos.contract.secType = "FUT"
    fut_pos.contract.symbol = "ES"
    fut_pos.contract.lastTradeDateOrContractMonth = "202406"
    fut_pos.contract.localSymbol = "ESM4"
    fut_pos.contract.multiplier = 50
    mock_conn.ib.portfolio.return_value = [fut_pos]
    found = broker.get_futures_position("ES", "202406")
    assert found is not None
    assert found["quantity"] == 1
    missing = broker.get_futures_position("ES", "202409")
    assert missing is None


# --- margin status --------------------------------------------------------

def test_audit_log_property_lazy_inits():
    """broker.audit_log returns the safeguard's AuditLog, lazy-constructed."""
    broker = IBKRFuturesBroker(config=IBKRConfig(port=4002))
    assert broker._audit_log is None
    al = broker.audit_log
    assert al is not None
    from src.trading.futures.audit_log import AuditLog
    assert isinstance(al, AuditLog)


def test_get_latest_trade_returns_dict_shape():
    """get_latest_trade extracts last/bid/ask/close from ticker, defaults to 0.0."""
    broker, mock_conn = _mk_broker_with_mock_conn()
    # Mock contract qualification
    fake_contract = MagicMock()
    fake_contract.localSymbol = "MESM4"
    fake_contract.symbol = "MES"
    mock_conn.ib.qualifyContracts.return_value = [fake_contract]
    # Mock the snapshot
    fake_ticker = MagicMock()
    fake_ticker.last = 5800.0
    fake_ticker.bid = 5799.75
    fake_ticker.ask = 5800.25
    fake_ticker.close = 5750.0
    mock_conn.run_sync.return_value = fake_ticker

    snap = broker.get_latest_trade("MES", "202406")
    assert snap["price"] == 5800.0
    assert snap["bid"] == 5799.75
    assert snap["ask"] == 5800.25
    assert snap["close"] == 5750.0
    assert snap["raw_symbol"] == "MESM4"


def test_get_latest_trade_handles_nan():
    """When ticker fields are NaN (market closed), all numeric fields are 0.0."""
    broker, mock_conn = _mk_broker_with_mock_conn()
    fake_contract = MagicMock()
    fake_contract.localSymbol = "MESM4"
    fake_contract.symbol = "MES"
    mock_conn.ib.qualifyContracts.return_value = [fake_contract]
    fake_ticker = MagicMock()
    fake_ticker.last = float("nan")
    fake_ticker.bid = float("nan")
    fake_ticker.ask = float("nan")
    fake_ticker.close = float("nan")
    mock_conn.run_sync.return_value = fake_ticker

    snap = broker.get_latest_trade("MES", "202406")
    assert snap["price"] == 0.0
    assert snap["bid"] == 0.0
    assert snap["ask"] == 0.0
    assert snap["close"] == 0.0


def test_get_margin_status_parses_account_values():
    broker, mock_conn = _mk_broker_with_mock_conn()
    mock_conn.ib.accountValues.return_value = [
        MagicMock(tag="NetLiquidation", value="100000.0", currency="USD"),
        MagicMock(tag="AvailableFunds", value="60000.0", currency="USD"),
        MagicMock(tag="InitMarginReq", value="25000.0", currency="USD"),
        MagicMock(tag="MaintMarginReq", value="20000.0", currency="USD"),
        MagicMock(tag="TotalCashValue", value="40000.0", currency="USD"),
        # Non-USD currency must be ignored
        MagicMock(tag="NetLiquidation", value="999999.0", currency="EUR"),
    ]
    s = broker.get_margin_status()
    assert s["net_liquidation"] == 100000.0
    assert s["available_funds"] == 60000.0
    assert s["initial_margin"] == 25000.0
    assert s["maintenance_margin"] == 20000.0
    assert s["free_cash"] == 40000.0
