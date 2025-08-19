from alpaca.data import StockHistoricalDataClient
from alpaca.data import StockBarsRequest
from alpaca.data import TimeFrame

import datetime

from api_key import API_KEY, API_SECRET

def fetch_data(start_date_str, end_date_str):
    if not API_KEY or not API_SECRET:
        raise ValueError("API keys not found. Check your .env file.")
    
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    start_date_obj = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

    request_params = StockBarsRequest(
                        symbol_or_symbols=["AAPL"],
                        timeframe=TimeFrame.Minute,
                        start=start_date_obj,
                        end=end_date_obj
                   )

    data = data_client.get_stock_bars(request_params).df