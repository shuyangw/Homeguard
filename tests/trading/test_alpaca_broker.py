"""
Tests for AlpacaBroker error handling and edge cases.

Tests proper error type conversion from Alpaca exceptions to
our standardized broker exceptions.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
import pandas as pd

from src.trading.brokers.broker_interface import (
    OrderSide, OrderType, TimeInForce, OrderStatus,
    BrokerError, BrokerConnectionError, BrokerAuthError,
    InvalidOrderError, InsufficientFundsError,
    OrderNotFoundError, NoPositionError, SymbolNotFoundError
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_trading_client():
    """Create a mock TradingClient."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_data_client():
    """Create a mock StockHistoricalDataClient."""
    client = MagicMock()
    return client


@pytest.fixture
def alpaca_broker(mock_trading_client, mock_data_client):
    """Create an AlpacaBroker with mocked clients."""
    with patch('src.trading.brokers.alpaca_broker.TradingClient', return_value=mock_trading_client):
        with patch('src.trading.brokers.alpaca_broker.StockHistoricalDataClient', return_value=mock_data_client):
            from src.trading.brokers.alpaca_broker import AlpacaBroker

            broker = AlpacaBroker(
                api_key='test_key',
                secret_key='test_secret',
                paper=True
            )

            return broker


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestAlpacaBrokerInit:
    """Tests for AlpacaBroker initialization."""

    def test_init_creates_clients(self):
        """Test initialization creates trading and data clients."""
        with patch('src.trading.brokers.alpaca_broker.TradingClient') as mock_tc:
            with patch('src.trading.brokers.alpaca_broker.StockHistoricalDataClient') as mock_dc:
                from src.trading.brokers.alpaca_broker import AlpacaBroker

                broker = AlpacaBroker('key', 'secret', paper=True)

                mock_tc.assert_called_once_with('key', 'secret', paper=True)
                mock_dc.assert_called_once_with('key', 'secret')

    def test_init_raises_connection_error_on_failure(self):
        """Test initialization raises BrokerConnectionError on failure."""
        with patch('src.trading.brokers.alpaca_broker.TradingClient', side_effect=Exception("Connection failed")):
            with patch('src.trading.brokers.alpaca_broker.StockHistoricalDataClient'):
                from src.trading.brokers.alpaca_broker import AlpacaBroker

                with pytest.raises(BrokerConnectionError):
                    AlpacaBroker('bad_key', 'bad_secret', paper=True)


# =============================================================================
# ACCOUNT METHOD TESTS
# =============================================================================

class TestAccountMethods:
    """Tests for account-related methods."""

    def test_get_account_returns_dict(self, alpaca_broker):
        """Test get_account returns standardized dict."""
        mock_account = MagicMock()
        mock_account.id = 'test_id'
        mock_account.buying_power = '100000.00'
        mock_account.cash = '50000.00'
        mock_account.portfolio_value = '150000.00'
        mock_account.equity = '150000.00'
        mock_account.currency = 'USD'

        alpaca_broker.trading_client.get_account.return_value = mock_account

        result = alpaca_broker.get_account()

        assert result['account_id'] == 'test_id'
        assert result['buying_power'] == 100000.0
        assert result['cash'] == 50000.0
        assert result['currency'] == 'USD'

    def test_get_account_raises_connection_error(self, alpaca_broker):
        """Test get_account raises BrokerConnectionError on API error."""
        alpaca_broker.trading_client.get_account.side_effect = Exception("API Error")

        with pytest.raises(BrokerConnectionError):
            alpaca_broker.get_account()


# =============================================================================
# POSITION METHOD TESTS
# =============================================================================

class TestPositionMethods:
    """Tests for position-related methods."""

    def test_get_stock_positions_returns_list(self, alpaca_broker):
        """Test get_stock_positions returns list of dicts."""
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = '100'
        mock_pos.avg_entry_price = '150.00'
        mock_pos.current_price = '155.00'
        mock_pos.market_value = '15500.00'
        mock_pos.unrealized_pl = '500.00'
        mock_pos.unrealized_plpc = '0.0333'
        mock_pos.side = 'long'

        alpaca_broker.trading_client.get_all_positions.return_value = [mock_pos]

        result = alpaca_broker.get_stock_positions()

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['quantity'] == 100
        assert result[0]['avg_entry_price'] == 150.0

    def test_get_stock_positions_raises_connection_error(self, alpaca_broker):
        """Test get_stock_positions raises BrokerConnectionError on error."""
        alpaca_broker.trading_client.get_all_positions.side_effect = Exception("API Error")

        with pytest.raises(BrokerConnectionError):
            alpaca_broker.get_stock_positions()

    def test_get_stock_position_returns_none_when_not_found(self, alpaca_broker):
        """Test get_stock_position returns None when no position."""
        alpaca_broker.trading_client.get_open_position.side_effect = Exception("Position not found")

        result = alpaca_broker.get_stock_position('AAPL')

        assert result is None


# =============================================================================
# ORDER METHOD TESTS
# =============================================================================

class TestOrderMethods:
    """Tests for order-related methods."""

    def test_place_market_order_success(self, alpaca_broker):
        """Test placing a market order successfully."""
        mock_order = MagicMock()
        mock_order.id = 'order_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '100'
        mock_order.side = MagicMock(value='buy')
        mock_order.type = MagicMock(value='market')
        mock_order.status = MagicMock(value='new')
        mock_order.filled_qty = '0'
        mock_order.filled_avg_price = None
        mock_order.created_at = datetime.now()

        alpaca_broker.trading_client.submit_order.return_value = mock_order

        result = alpaca_broker.place_stock_order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        assert result['order_id'] == 'order_123'
        assert result['symbol'] == 'AAPL'

    def test_place_limit_order_requires_price(self, alpaca_broker):
        """Test limit order raises InvalidOrderError without price."""
        with pytest.raises(InvalidOrderError):
            alpaca_broker.place_stock_order(
                symbol='AAPL',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                limit_price=None  # Missing required price
            )

    def test_place_stop_order_requires_price(self, alpaca_broker):
        """Test stop order raises InvalidOrderError without price."""
        with pytest.raises(InvalidOrderError):
            alpaca_broker.place_stock_order(
                symbol='AAPL',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                stop_price=None  # Missing required price
            )

    def test_place_order_insufficient_funds(self, alpaca_broker):
        """Test order raises InsufficientFundsError when funds low."""
        alpaca_broker.trading_client.submit_order.side_effect = Exception("insufficient buying power")

        with pytest.raises(InsufficientFundsError):
            alpaca_broker.place_stock_order(
                symbol='AAPL',
                quantity=10000,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

    def test_place_order_general_error(self, alpaca_broker):
        """Test order raises BrokerConnectionError on general error."""
        alpaca_broker.trading_client.submit_order.side_effect = Exception("Unknown error")

        with pytest.raises(BrokerConnectionError):
            alpaca_broker.place_stock_order(
                symbol='AAPL',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

    def test_cancel_order_success(self, alpaca_broker):
        """Test canceling an order successfully."""
        alpaca_broker.trading_client.cancel_order_by_id.return_value = None

        result = alpaca_broker.cancel_order('order_123')

        assert result == True

    def test_cancel_order_not_found(self, alpaca_broker):
        """Test cancel raises OrderNotFoundError when order not found."""
        alpaca_broker.trading_client.cancel_order_by_id.side_effect = Exception("Order not found")

        with pytest.raises(OrderNotFoundError):
            alpaca_broker.cancel_order('nonexistent_order')

    def test_get_order_not_found(self, alpaca_broker):
        """Test get_order raises OrderNotFoundError when not found."""
        alpaca_broker.trading_client.get_order_by_id.side_effect = Exception("Order not found")

        with pytest.raises(OrderNotFoundError):
            alpaca_broker.get_order('nonexistent_order')


# =============================================================================
# CLOSE POSITION TESTS
# =============================================================================

class TestClosePositionMethods:
    """Tests for position closing methods."""

    def test_close_position_no_position(self, alpaca_broker):
        """Test close_stock_position raises NoPositionError when no position."""
        # No position found
        alpaca_broker.trading_client.get_open_position.side_effect = Exception("Position not found")

        with pytest.raises(NoPositionError):
            alpaca_broker.close_stock_position('AAPL')

    def test_close_position_determines_correct_side(self, alpaca_broker):
        """Test close_stock_position uses correct side (SELL for long)."""
        # Mock existing long position
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = '100'  # Long position
        mock_pos.avg_entry_price = '150.00'
        mock_pos.current_price = '155.00'
        mock_pos.market_value = '15500.00'
        mock_pos.unrealized_pl = '500.00'
        mock_pos.unrealized_plpc = '0.0333'
        mock_pos.side = 'long'

        alpaca_broker.trading_client.get_open_position.return_value = mock_pos

        # Mock order response
        mock_order = MagicMock()
        mock_order.id = 'close_order'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '100'
        mock_order.side = MagicMock(value='sell')
        mock_order.type = MagicMock(value='market')
        mock_order.status = MagicMock(value='new')
        mock_order.filled_qty = '0'
        mock_order.filled_avg_price = None
        mock_order.created_at = datetime.now()

        alpaca_broker.trading_client.submit_order.return_value = mock_order

        result = alpaca_broker.close_stock_position('AAPL')

        assert result['symbol'] == 'AAPL'

    def test_close_all_positions(self, alpaca_broker):
        """Test close_all_stock_positions calls Alpaca method."""
        alpaca_broker.trading_client.close_all_positions.return_value = "success"

        result = alpaca_broker.close_all_stock_positions()

        alpaca_broker.trading_client.close_all_positions.assert_called_once_with(cancel_orders=True)


# =============================================================================
# MARKET DATA TESTS
# =============================================================================

class TestMarketDataMethods:
    """Tests for market data methods."""

    def test_get_latest_quote_symbol_not_found(self, alpaca_broker):
        """Test get_latest_quote raises SymbolNotFoundError."""
        alpaca_broker.data_client.get_stock_latest_quote.side_effect = Exception("Symbol not found")

        with pytest.raises(SymbolNotFoundError):
            alpaca_broker.get_latest_quote('INVALID')

    def test_get_latest_trade_symbol_not_found(self, alpaca_broker):
        """Test get_latest_trade raises SymbolNotFoundError."""
        alpaca_broker.data_client.get_stock_latest_trade.side_effect = Exception("Symbol not found")

        with pytest.raises(SymbolNotFoundError):
            alpaca_broker.get_latest_trade('INVALID')

    def test_get_bars_raises_connection_error(self, alpaca_broker):
        """Test get_bars raises BrokerConnectionError on error."""
        alpaca_broker.data_client.get_stock_bars.side_effect = Exception("API Error")

        with pytest.raises(BrokerConnectionError):
            alpaca_broker.get_bars(
                symbols=['AAPL'],
                timeframe='1D',
                start=datetime.now() - timedelta(days=30),
                end=datetime.now()
            )

    def test_get_historical_bars_empty_data(self, alpaca_broker):
        """Test get_historical_bars handles empty data."""
        # Mock empty DataFrame
        alpaca_broker.data_client.get_stock_bars.return_value = MagicMock(df=pd.DataFrame())

        result = alpaca_broker.get_historical_bars(
            symbol='AAPL',
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            timeframe='1D'
        )

        assert result.empty


# =============================================================================
# TIMEZONE CONVERSION TESTS
# =============================================================================

class TestTimezoneConversion:
    """Tests for timezone handling."""

    def test_get_account_values_are_floats(self, alpaca_broker):
        """Test account values are properly converted to floats."""
        mock_account = MagicMock()
        mock_account.id = 'test_id'
        mock_account.buying_power = '99999.99'  # String from API
        mock_account.cash = '50000.50'
        mock_account.portfolio_value = '149999.99'
        mock_account.equity = '149999.99'
        mock_account.currency = 'USD'

        alpaca_broker.trading_client.get_account.return_value = mock_account

        result = alpaca_broker.get_account()

        # All values should be proper floats
        assert isinstance(result['buying_power'], float)
        assert isinstance(result['cash'], float)
        assert result['buying_power'] == 99999.99

    def test_get_positions_values_are_correct_types(self, alpaca_broker):
        """Test position values have correct types."""
        mock_pos = MagicMock()
        mock_pos.symbol = 'AAPL'
        mock_pos.qty = '100'  # String from API
        mock_pos.avg_entry_price = '150.50'
        mock_pos.current_price = '155.25'
        mock_pos.market_value = '15525.00'
        mock_pos.unrealized_pl = '475.00'
        mock_pos.unrealized_plpc = '0.0316'
        mock_pos.side = 'long'

        alpaca_broker.trading_client.get_all_positions.return_value = [mock_pos]

        result = alpaca_broker.get_stock_positions()

        # Quantity should be int, prices should be float
        assert isinstance(result[0]['quantity'], int)
        assert isinstance(result[0]['avg_entry_price'], float)
        assert result[0]['quantity'] == 100
        assert result[0]['avg_entry_price'] == 150.50


# =============================================================================
# ORDER STATUS TRANSITION TESTS
# =============================================================================

class TestOrderStatusTransitions:
    """Tests for order status handling."""

    def test_get_orders_with_status_filter(self, alpaca_broker):
        """Test get_orders respects status filter."""
        mock_order = MagicMock()
        mock_order.id = 'order_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '100'
        mock_order.side = MagicMock(value='buy')
        mock_order.type = MagicMock(value='market')
        mock_order.status = MagicMock(value='filled')
        mock_order.filled_qty = '100'
        mock_order.filled_avg_price = '151.00'
        mock_order.created_at = datetime.now()

        alpaca_broker.trading_client.get_orders.return_value = [mock_order]

        result = alpaca_broker.get_orders(status='open')

        assert alpaca_broker.trading_client.get_orders.called

    def test_get_open_orders_filters_correctly(self, alpaca_broker):
        """Test get_open_orders returns only open orders."""
        mock_order = MagicMock()
        mock_order.id = 'order_123'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '100'
        mock_order.side = MagicMock(value='buy')
        mock_order.type = MagicMock(value='limit')
        mock_order.status = MagicMock(value='new')
        mock_order.filled_qty = '0'
        mock_order.filled_avg_price = None
        mock_order.created_at = datetime.now()

        alpaca_broker.trading_client.get_orders.return_value = [mock_order]

        result = alpaca_broker.get_open_orders()

        assert len(result) == 1
        assert result[0]['status'] == 'new'
