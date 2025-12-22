"""
Pytest fixtures for yfinance module tests.
"""

import pytest
from typing import Dict


@pytest.fixture
def sample_yfinance_info() -> Dict:
    """Sample yfinance Ticker.info dict for AAPL."""
    return {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "marketCap": 3000000000000,  # $3T
        "enterpriseValue": 3100000000000,
        "sharesOutstanding": 15500000000,
        "trailingPE": 28.5,
        "forwardPE": 25.2,
        "pegRatio": 2.1,
        "priceToBook": 45.3,
        "priceToSalesTrailing12Months": 7.8,
        "trailingEps": 6.15,
        "forwardEps": 6.85,
        "earningsQuarterlyGrowth": 0.12,
        "revenueGrowth": 0.08,
        "profitMargins": 0.25,
        "operatingMargins": 0.30,
        "grossMargins": 0.43,
        "returnOnEquity": 0.147,
        "returnOnAssets": 0.21,
        "debtToEquity": 195.5,
        "currentRatio": 0.94,
        "quickRatio": 0.85,
        "dividendYield": 0.0052,
        "dividendRate": 0.96,
        "payoutRatio": 0.15,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "exchange": "NMS",
        "beta": 1.27,
        "targetMeanPrice": 195.0,
        "targetHighPrice": 250.0,
        "targetLowPrice": 150.0,
        "recommendationKey": "buy",
        "shortRatio": 1.5,
        "shortPercentOfFloat": 0.007,
    }


@pytest.fixture
def sample_yfinance_info_minimal() -> Dict:
    """Minimal yfinance info with only essential fields."""
    return {
        "symbol": "MSFT",
        "shortName": "Microsoft Corp",
        "marketCap": 2800000000000,
        "sector": "Technology",
    }


@pytest.fixture
def sample_yfinance_info_dividend() -> Dict:
    """Sample yfinance info for a high dividend stock."""
    return {
        "symbol": "JNJ",
        "shortName": "Johnson & Johnson",
        "marketCap": 400000000000,
        "trailingPE": 15.2,
        "dividendYield": 0.029,
        "dividendRate": 4.76,
        "payoutRatio": 0.45,
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
    }


@pytest.fixture
def sample_symbols() -> list:
    """Sample list of symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
