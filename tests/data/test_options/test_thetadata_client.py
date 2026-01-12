"""Tests for ThetaData client."""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.options.thetadata_client import (
    ThetaDataClient,
    ThetaDataError,
    ThetaTerminalNotRunning,
    ThetaDataRateLimited,
)
from src.data.options.options_schema import OptionsChain, OptionRight


class TestThetaDataClientConnection:
    """Tests for client connection handling."""

    def test_default_ports(self):
        """Test default port configuration."""
        client = ThetaDataClient()

        assert client.host == "127.0.0.1"
        assert client.port_v2 == 25510
        assert client.port_v3 == 25503

    def test_custom_ports(self):
        """Test custom port configuration."""
        client = ThetaDataClient(host="192.168.1.1", port_v2=25511, port_v3=25504)

        assert client.host == "192.168.1.1"
        assert client.port_v2 == 25511
        assert client.port_v3 == 25504

    def test_check_connection_success(self):
        """Test successful connection check."""
        client = ThetaDataClient()

        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        client._session = mock_session

        assert client.check_connection() is True

    def test_check_connection_failure(self):
        """Test failed connection check."""
        import requests

        client = ThetaDataClient()

        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError()

        client._session = mock_session

        assert client.check_connection() is False


class TestThetaDataClientExpirations:
    """Tests for expiration listing."""

    @patch.object(ThetaDataClient, '_make_request')
    def test_list_expirations(self, mock_request):
        """Test listing expirations."""
        mock_request.return_value = {
            "response": [20240119, 20240216, 20240315]
        }

        client = ThetaDataClient()
        expirations = client.list_expirations("SPY")

        assert len(expirations) == 3
        assert expirations[0] == date(2024, 1, 19)
        assert expirations[1] == date(2024, 2, 16)
        assert expirations[2] == date(2024, 3, 15)

    @patch.object(ThetaDataClient, '_make_request')
    def test_list_expirations_with_filter(self, mock_request):
        """Test filtering expirations by date range."""
        mock_request.return_value = {
            "response": [20240119, 20240216, 20240315, 20240419]
        }

        client = ThetaDataClient()
        expirations = client.list_expirations(
            "SPY",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 3, 31),
        )

        assert len(expirations) == 2
        assert expirations[0] == date(2024, 2, 16)
        assert expirations[1] == date(2024, 3, 15)


class TestThetaDataClientStrikes:
    """Tests for strike listing."""

    @patch.object(ThetaDataClient, '_make_request')
    def test_list_strikes(self, mock_request):
        """Test listing strike prices."""
        # ThetaData uses 1/10th cent format: 170000 = $170.00
        mock_request.return_value = {
            "response": [470000, 475000, 480000]
        }

        client = ThetaDataClient()
        strikes = client.list_strikes("SPY", date(2024, 1, 19))

        assert len(strikes) == 3
        assert strikes[0] == 470.0
        assert strikes[1] == 475.0
        assert strikes[2] == 480.0


class TestThetaDataClientEOD:
    """Tests for EOD data fetching."""

    @patch.object(ThetaDataClient, '_make_request')
    def test_get_eod_data(self, mock_request):
        """Test getting EOD data for a contract."""
        mock_request.return_value = {
            "response": [
                {
                    "open_interest": 5000,
                    "volume": 1000,
                    "bid": 2.50,
                    "ask": 2.60,
                    "underlying_price": 475.0,
                }
            ]
        }

        client = ThetaDataClient()
        data = client.get_eod_data(
            symbol="SPY",
            expiry=date(2024, 1, 19),
            strike=475.0,
            right="C",
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        assert len(data) == 1
        assert data[0]["open_interest"] == 5000


class TestThetaDataClientChain:
    """Tests for options chain fetching."""

    @patch.object(ThetaDataClient, 'list_strikes')
    @patch.object(ThetaDataClient, 'get_eod_data')
    def test_get_options_chain(self, mock_eod, mock_strikes):
        """Test getting complete options chain."""
        # Mock strikes
        mock_strikes.return_value = [470.0, 475.0, 480.0]

        # Mock EOD data for each contract
        mock_eod.return_value = [{
            "open_interest": 5000,
            "volume": 1000,
            "bid": 2.50,
            "ask": 2.60,
            "underlying_price": 475.0,
        }]

        client = ThetaDataClient()
        chain = client.get_options_chain(
            symbol="SPY",
            snapshot_date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
        )

        assert isinstance(chain, OptionsChain)
        assert chain.symbol == "SPY"
        # Should have 6 contracts (3 strikes x 2 rights)
        assert len(chain.contracts) == 6

    @patch.object(ThetaDataClient, 'list_strikes')
    def test_get_options_chain_no_strikes(self, mock_strikes):
        """Test getting chain when no strikes available."""
        mock_strikes.return_value = []

        client = ThetaDataClient()
        chain = client.get_options_chain(
            symbol="SPY",
            snapshot_date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
        )

        assert chain.contracts == []
        assert chain.underlying_price == 0.0


class TestThetaDataClientMonthlyFilter:
    """Tests for monthly expiration filtering."""

    def test_filter_monthly_expirations(self):
        """Test filtering to monthly expirations only."""
        client = ThetaDataClient()

        expirations = [
            date(2024, 1, 12),  # Weekly
            date(2024, 1, 19),  # Monthly (3rd Friday)
            date(2024, 1, 26),  # Weekly
            date(2024, 2, 16),  # Monthly (3rd Friday)
            date(2024, 2, 23),  # Weekly
        ]

        monthly = client._filter_monthly_expirations(expirations)

        assert len(monthly) == 2
        assert date(2024, 1, 19) in monthly
        assert date(2024, 2, 16) in monthly

    def test_filter_monthly_edge_cases(self):
        """Test edge cases in monthly filtering."""
        client = ThetaDataClient()

        # February 2024 - 3rd Friday is Feb 16
        feb_dates = [
            date(2024, 2, 2),   # 1st Friday
            date(2024, 2, 9),   # 2nd Friday
            date(2024, 2, 16),  # 3rd Friday (monthly)
            date(2024, 2, 23),  # 4th Friday
        ]

        monthly = client._filter_monthly_expirations(feb_dates)

        assert len(monthly) == 1
        assert monthly[0] == date(2024, 2, 16)


class TestThetaDataClientDataFrame:
    """Tests for DataFrame conversion."""

    def test_to_dataframe(self):
        """Test converting chain to DataFrame."""
        from src.data.options.options_schema import OptionSnapshot, OptionsChain

        contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=475.0,
                option_type=OptionRight.CALL,
                open_interest=5000,
                volume=1000,
                delta=0.5,
                gamma=0.02,
            ),
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=475.0,
                option_type=OptionRight.PUT,
                open_interest=3000,
                volume=500,
                delta=-0.5,
                gamma=0.02,
            ),
        ]

        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=contracts,
        )

        client = ThetaDataClient()
        df = client.to_dataframe(chain)

        assert len(df) == 2
        assert "symbol" in df.columns
        assert "strike" in df.columns
        assert "delta" in df.columns
        assert "gamma" in df.columns

    def test_to_dataframe_empty(self):
        """Test converting empty chain to DataFrame."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=[],
        )

        client = ThetaDataClient()
        df = client.to_dataframe(chain)

        assert len(df) == 0


class TestThetaDataClientErrors:
    """Tests for error handling."""

    @patch('requests.Session.get')
    def test_connection_error(self, mock_get):
        """Test connection error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        client = ThetaDataClient()

        with pytest.raises(ThetaTerminalNotRunning):
            client._make_request("http://localhost:25510/test", {})

    @patch('requests.Session.get')
    def test_rate_limit_error(self, mock_get):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_get.return_value = mock_response

        client = ThetaDataClient()

        with pytest.raises(ThetaDataRateLimited):
            client._make_request("http://localhost:25510/test", {})

    @patch('requests.Session.get')
    def test_timeout_error(self, mock_get):
        """Test timeout error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        client = ThetaDataClient()

        with pytest.raises(ThetaDataError):
            client._make_request("http://localhost:25510/test", {})
