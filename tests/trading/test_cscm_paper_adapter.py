"""
Unit tests for CSCM Paper Trading Adapter.

Tests:
- Initialization with Alpaca paper trading
- Data provider override (Binance with fallback)
- Rebalance day configuration (Monday)
- State persistence
- Portfolio value tracking
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile


class TestCSCMPaperAdapterInitialization:
    """Test CSCM paper adapter initialization."""

    @patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker')
    @patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback')
    @patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state')
    def test_initializes_with_alpaca_paper(
        self, mock_load_state, mock_data_provider, mock_broker
    ):
        """Test that adapter initializes with Alpaca paper trading."""
        from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter

        mock_broker.return_value = Mock()
        mock_data_provider.return_value = Mock()

        adapter = CSCMPaperAdapter()

        mock_broker.assert_called_once_with(paper=True)
        assert adapter.paper is True

    @patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker')
    @patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback')
    @patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state')
    def test_default_rebalance_day_is_monday(
        self, mock_load_state, mock_data_provider, mock_broker
    ):
        """Test that default rebalance day is Monday."""
        from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter

        mock_broker.return_value = Mock()
        mock_data_provider.return_value = Mock()

        adapter = CSCMPaperAdapter()

        assert adapter.rebalance_day == 'monday'

    @patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker')
    @patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback')
    @patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state')
    def test_uses_binance_data_provider(
        self, mock_load_state, mock_data_provider, mock_broker
    ):
        """Test that adapter uses Binance data provider with fallback."""
        from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter

        mock_broker.return_value = Mock()
        mock_provider_instance = Mock()
        mock_data_provider.return_value = mock_provider_instance

        adapter = CSCMPaperAdapter()

        mock_data_provider.assert_called_once()
        assert adapter._data_provider == mock_provider_instance


class TestCSCMPaperAdapterDataFetching:
    """Test data fetching behavior."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker') as mock_broker, \
             patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback') as mock_provider, \
             patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state'):

            mock_broker.return_value = Mock()
            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance

            from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
            adapter = CSCMPaperAdapter()
            adapter._data_provider = mock_provider_instance

            return adapter

    def test_fetch_historical_data_uses_binance(self, adapter):
        """Test that historical data uses Binance provider."""
        import pandas as pd

        mock_data = {'BTC/USD': pd.DataFrame({'close': [50000, 51000]})}
        adapter._data_provider.get_historical_bars_batch.return_value = mock_data

        result = adapter._fetch_historical_data()

        adapter._data_provider.get_historical_bars_batch.assert_called_once()
        assert result == mock_data

    def test_get_current_prices_uses_binance(self, adapter):
        """Test that current prices use Binance provider."""
        adapter._data_provider.get_current_prices.return_value = {
            'BTC/USD': 50000.0,
            'ETH/USD': 3000.0
        }

        result = adapter._get_current_prices()

        adapter._data_provider.get_current_prices.assert_called_once_with(adapter.universe)
        assert result['BTC/USD'] == 50000.0


class TestCSCMPaperAdapterStatus:
    """Test status reporting."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker') as mock_broker, \
             patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback') as mock_provider, \
             patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state'):

            mock_broker_instance = Mock()
            mock_broker.return_value = mock_broker_instance

            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance

            from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
            adapter = CSCMPaperAdapter()
            adapter._broker = mock_broker_instance
            adapter._data_provider = mock_provider_instance

            return adapter

    def test_get_status_includes_paper_mode(self, adapter):
        """Test that status includes paper mode indicator."""
        # Mock dependencies
        adapter._data_provider.get_historical_bars_batch.return_value = {}
        adapter._broker.get_crypto_positions.return_value = []
        adapter._broker.get_crypto_balance.return_value = {'available': 10000.0}

        # Mock signals
        adapter.signals.update_historical_data = Mock()
        adapter.signals.get_risk_signals = Mock(return_value=Mock(
            regime='bullish',
            btc_price=50000,
            btc_sma=48000,
            is_rebalance_day=False,
            top_symbols=['ETH/USD', 'SOL/USD'],
            trailing_stop_triggered=False,
            current_drawdown=0.05,
            peak_value=11000.0
        ))
        adapter.signals.get_target_positions = Mock(return_value={})

        status = adapter.get_status()

        assert status['mode'] == 'paper'
        assert status['data_source'] == 'Binance (Alpaca fallback)'
        assert status['execution'] == 'Alpaca Paper'


class TestCSCMPaperAdapterStatePersistence:
    """Test state persistence behavior."""

    def test_uses_paper_specific_cache_file(self):
        """Test that paper adapter uses separate cache file."""
        with patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker'), \
             patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback'), \
             patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state'):

            from src.trading.adapters.cscm_paper_adapter import (
                CSCMPaperAdapter, PAPER_CACHE_FILE
            )

            adapter = CSCMPaperAdapter()

            assert adapter._cache_file == PAPER_CACHE_FILE
            assert 'paper' in str(PAPER_CACHE_FILE).lower()


class TestCSCMPaperAdapterPortfolioTracker:
    """Test portfolio tracker integration."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker') as mock_broker, \
             patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback') as mock_provider, \
             patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state'):

            mock_broker.return_value = Mock()
            mock_provider.return_value = Mock()

            from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
            return CSCMPaperAdapter()

    def test_get_portfolio_tracker_lazy_init(self, adapter):
        """Test that portfolio tracker is lazily initialized."""
        assert adapter._portfolio_tracker is None

        with patch('src.trading.portfolio.tracker.PortfolioTracker') as mock_tracker:
            mock_tracker.return_value = Mock()

            tracker = adapter.get_portfolio_tracker()

            assert tracker is not None
            mock_tracker.assert_called_once()

    def test_get_portfolio_value(self, adapter):
        """Test getting portfolio value."""
        adapter._broker = Mock()
        adapter._broker.get_crypto_positions.return_value = [
            {'symbol': 'BTC/USD', 'quantity': Decimal('0.1')}
        ]
        adapter._broker.get_crypto_balance.return_value = {'available': 5000.0}
        adapter._broker.get_crypto_quote.return_value = {'last': 50000.0}

        value = adapter.get_portfolio_value()

        assert value == 10000.0  # 5000 cash + 0.1 * 50000


class TestCSCMPaperAdapterFactory:
    """Test factory function."""

    @patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker')
    @patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback')
    @patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state')
    def test_create_paper_adapter_with_defaults(
        self, mock_load_state, mock_data_provider, mock_broker
    ):
        """Test creating adapter with default parameters."""
        mock_broker.return_value = Mock()
        mock_data_provider.return_value = Mock()

        from src.trading.adapters.cscm_paper_adapter import create_paper_adapter

        adapter = create_paper_adapter()

        assert adapter.top_n == 5
        assert adapter.rebalance_day == 'monday'
        assert adapter.trailing_stop_pct == 0.25

    @patch('src.trading.adapters.cscm_paper_adapter.AlpacaCryptoBroker')
    @patch('src.trading.adapters.cscm_paper_adapter.CryptoDataProviderWithFallback')
    @patch('src.trading.adapters.cscm_paper_adapter.CSCMLiveAdapter._load_state')
    def test_create_paper_adapter_with_overrides(
        self, mock_load_state, mock_data_provider, mock_broker
    ):
        """Test creating adapter with custom parameters."""
        mock_broker.return_value = Mock()
        mock_data_provider.return_value = Mock()

        from src.trading.adapters.cscm_paper_adapter import create_paper_adapter

        adapter = create_paper_adapter(top_n=3, rebalance_day='tuesday')

        assert adapter.top_n == 3
        assert adapter.rebalance_day == 'tuesday'
