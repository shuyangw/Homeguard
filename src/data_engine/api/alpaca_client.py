"""
Alpaca API client for fetching stock market data.
"""

from alpaca.data import StockHistoricalDataClient
from alpaca.data import StockBarsRequest
from alpaca.data import TimeFrame
from alpaca.data.enums import DataFeed
import datetime
import threading

from api_key import API_KEY, API_SECRET


class AlpacaClient:
    """
    Client for fetching historical stock data from Alpaca API.
    """

    def __init__(self, api_key=None, api_secret=None, feed=None):
        """
        Initialize the Alpaca client.

        Args:
            api_key: Alpaca API key (defaults to API_KEY from api_key module)
            api_secret: Alpaca API secret (defaults to API_SECRET from api_key module)
            feed: Data feed to use (DataFeed.SIP, DataFeed.IEX, DataFeed.OTC, DataFeed.DELAYED_SIP)
                  Defaults to None (uses Alpaca's default, typically SIP for paid plans)
        """
        self.api_key = api_key or API_KEY
        self.api_secret = api_secret or API_SECRET
        self.feed = feed

        if not self.api_key or not self.api_secret:
            raise ValueError("API keys not found. Check your .env file.")

        self.client = StockHistoricalDataClient(self.api_key, self.api_secret)

    def fetch_bars(self, symbol, start_date_str, end_date_str, timeframe=None, feed=None):
        """
        Fetch stock bar data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date_str: Start date in 'YYYY-MM-DD' format
            end_date_str: End date in 'YYYY-MM-DD' format
            timeframe: TimeFrame object (default: TimeFrame.Minute)
            feed: Data feed to use (overrides client-level feed if specified)

        Returns:
            pandas DataFrame with stock data
        """
        start_date_obj = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date_obj = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

        timeframe_value = timeframe if timeframe is not None else TimeFrame.Minute
        data_feed = feed or self.feed

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe_value,  # type: ignore[arg-type]
            start=start_date_obj,
            end=end_date_obj,
            feed=data_feed
        )

        thread_id = threading.get_ident()
        feed_info = f" (feed: {data_feed})" if data_feed else ""
        print(f"[Thread-{thread_id}] Fetching data for {symbol} from {start_date_str} to {end_date_str}{feed_info}")

        data = self.client.get_stock_bars(request_params).df
        print(f"[Thread-{thread_id}] Fetched {len(data)} bars for {symbol}")

        return data
