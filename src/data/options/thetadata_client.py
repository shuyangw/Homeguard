"""
ThetaData REST API client for options data.

Fetches historical and real-time options data including:
- Options chains with Greeks
- Open interest
- EOD (end-of-day) historical data

Requires Theta Terminal to be running locally.
See: https://docs.thetadata.us/
"""

import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from requests.exceptions import RequestException
import pandas as pd
from dataclasses import asdict

from src.utils.logger import logger
from src.data.options.options_schema import (
    OptionSnapshot,
    OptionsChain,
    OptionRight,
)


class ThetaDataError(Exception):
    """Base exception for ThetaData API errors."""
    pass


class ThetaTerminalNotRunning(ThetaDataError):
    """Raised when Theta Terminal is not running."""
    pass


class ThetaDataRateLimited(ThetaDataError):
    """Raised when rate limited by API."""
    pass


class ThetaDataClient:
    """
    Client for ThetaData REST API.

    Fetches options chain data for GEX calculations.
    Requires Theta Terminal running locally.

    Usage:
        client = ThetaDataClient()
        chain = client.get_options_chain("SPY", date(2024, 1, 15), date(2024, 1, 19))

    Environment:
        THETADATA_HOST: API host (default: localhost)
        THETADATA_PORT_V2: Port for v2 API (default: 25510)
        THETADATA_PORT_V3: Port for v3 API (default: 25503)
    """

    # Default ports for Theta Terminal
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT_V2 = 25510  # Historical data
    DEFAULT_PORT_V3 = 25503  # Snapshots

    # Rate limiting
    REQUESTS_PER_SECOND = 10
    MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND

    def __init__(
        self,
        host: Optional[str] = None,
        port_v2: Optional[int] = None,
        port_v3: Optional[int] = None,
        timeout: int = 30,
    ):
        """
        Initialize ThetaData client.

        Args:
            host: API host (default: localhost or THETADATA_HOST env var)
            port_v2: Port for v2 API (historical)
            port_v3: Port for v3 API (snapshots)
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv("THETADATA_HOST", self.DEFAULT_HOST)
        self.port_v2 = port_v2 or int(os.getenv("THETADATA_PORT_V2", self.DEFAULT_PORT_V2))
        self.port_v3 = port_v3 or int(os.getenv("THETADATA_PORT_V3", self.DEFAULT_PORT_V3))
        self.timeout = timeout

        self._base_url_v2 = f"http://{self.host}:{self.port_v2}"
        self._base_url_v3 = f"http://{self.host}:{self.port_v3}"

        self._last_request_time = 0.0
        self._session = requests.Session()

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        url: str,
        params: Dict[str, Any],
        version: str = "v2"
    ) -> Dict[str, Any]:
        """
        Make API request with error handling.

        Args:
            url: Full URL endpoint
            params: Query parameters
            version: API version for base URL

        Returns:
            JSON response as dict

        Raises:
            ThetaTerminalNotRunning: If terminal not running
            ThetaDataRateLimited: If rate limited
            ThetaDataError: For other API errors
        """
        self._rate_limit()

        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                raise ThetaDataRateLimited(
                    f"Rate limited. Wait and retry. Response: {response.text}"
                )

            elif response.status_code == 503:
                raise ThetaTerminalNotRunning(
                    "Theta Terminal not running or not connected"
                )

            else:
                raise ThetaDataError(
                    f"API error {response.status_code}: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            raise ThetaTerminalNotRunning(
                f"Cannot connect to Theta Terminal at {self.host}. "
                "Ensure Theta Terminal is running."
            )

        except requests.exceptions.Timeout:
            raise ThetaDataError(f"Request timed out after {self.timeout}s")

    def check_connection(self) -> bool:
        """
        Check if Theta Terminal is running and responsive.

        Returns:
            True if connected, False otherwise
        """
        try:
            # Try a simple request to v2 API
            response = self._session.get(
                f"{self._base_url_v2}/",
                timeout=5,
            )
            return response.status_code in [200, 404]  # 404 OK, just means no endpoint
        except (ConnectionError, TimeoutError, RequestException):
            return False

    def list_expirations(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """
        List available option expirations for a symbol.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start_date: Filter expirations on or after this date
            end_date: Filter expirations on or before this date

        Returns:
            List of expiration dates
        """
        url = f"{self._base_url_v2}/v2/list/expirations"
        params = {
            "root": symbol.upper(),
        }

        try:
            data = self._make_request(url, params)
            expirations = []

            # Parse expiration dates from response
            if "response" in data:
                for exp_str in data["response"]:
                    exp_date = datetime.strptime(str(exp_str), "%Y%m%d").date()
                    expirations.append(exp_date)

            # Filter by date range
            if start_date:
                expirations = [e for e in expirations if e >= start_date]
            if end_date:
                expirations = [e for e in expirations if e <= end_date]

            return sorted(expirations)

        except ThetaDataError:
            raise
        except Exception as e:
            logger.error(f"Error listing expirations for {symbol}: {e}")
            raise ThetaDataError(f"Failed to list expirations: {e}")

    def list_strikes(
        self,
        symbol: str,
        expiry: date,
    ) -> List[float]:
        """
        List available strike prices for a symbol/expiration.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date

        Returns:
            List of strike prices
        """
        url = f"{self._base_url_v2}/v2/list/strikes"
        params = {
            "root": symbol.upper(),
            "exp": expiry.strftime("%Y%m%d"),
        }

        try:
            data = self._make_request(url, params)
            strikes = []

            if "response" in data:
                for strike_1000 in data["response"]:
                    # ThetaData uses 1/10th cent format: 170000 = $170.00
                    strikes.append(strike_1000 / 1000.0)

            return sorted(strikes)

        except ThetaDataError:
            raise
        except Exception as e:
            logger.error(f"Error listing strikes for {symbol} {expiry}: {e}")
            raise ThetaDataError(f"Failed to list strikes: {e}")

    def get_eod_data(
        self,
        symbol: str,
        expiry: date,
        strike: float,
        right: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Get historical EOD data for a single option contract.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strike: Strike price in dollars
            right: 'C' for call, 'P' for put
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of EOD records
        """
        url = f"{self._base_url_v2}/v2/hist/option/eod"

        # Convert strike to ThetaData format (1/10th cent)
        strike_1000 = int(strike * 1000)

        params = {
            "root": symbol.upper(),
            "exp": expiry.strftime("%Y%m%d"),
            "strike": strike_1000,
            "right": right.upper(),
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
        }

        try:
            data = self._make_request(url, params)
            return data.get("response", [])

        except ThetaDataError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting EOD for {symbol} {expiry} {strike} {right}: {e}"
            )
            raise ThetaDataError(f"Failed to get EOD data: {e}")

    def get_greeks_snapshot(
        self,
        symbol: str,
        expiry: date,
        strike: Optional[float] = None,
        right: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        Get current Greeks snapshot for options chain.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date, or '*' for all
            strike: Strike price, or None for all strikes
            right: 'call', 'put', or 'both'

        Returns:
            List of Greeks records
        """
        url = f"{self._base_url_v3}/v3/option/snapshot/greeks/all"

        params = {
            "symbol": symbol.upper(),
            "expiration": expiry.strftime("%Y-%m-%d") if isinstance(expiry, date) else expiry,
            "right": right.lower(),
            "format": "json",
        }

        if strike is not None:
            params["strike"] = f"{strike:.2f}"

        try:
            data = self._make_request(url, params, version="v3")
            return data.get("response", [])

        except ThetaDataError:
            raise
        except Exception as e:
            logger.error(f"Error getting Greeks for {symbol} {expiry}: {e}")
            raise ThetaDataError(f"Failed to get Greeks: {e}")

    def get_open_interest_snapshot(
        self,
        symbol: str,
        expiry: date,
        strike: Optional[float] = None,
        right: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        Get current open interest snapshot.

        Note: OI is reported around 06:30 ET and reflects prior day's OI.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strike: Strike price, or None for all
            right: 'call', 'put', or 'both'

        Returns:
            List of OI records
        """
        url = f"{self._base_url_v3}/v3/option/snapshot/open_interest"

        params = {
            "symbol": symbol.upper(),
            "expiration": expiry.strftime("%Y-%m-%d"),
            "right": right.lower(),
            "format": "json",
        }

        if strike is not None:
            params["strike"] = f"{strike:.2f}"

        try:
            data = self._make_request(url, params, version="v3")
            return data.get("response", [])

        except ThetaDataError:
            raise
        except Exception as e:
            logger.error(f"Error getting OI for {symbol} {expiry}: {e}")
            raise ThetaDataError(f"Failed to get open interest: {e}")

    def get_options_chain(
        self,
        symbol: str,
        snapshot_date: date,
        expiry: date,
        strike_range_pct: float = 0.20,
        underlying_price: Optional[float] = None,
    ) -> OptionsChain:
        """
        Get complete options chain for a symbol/date/expiry.

        This fetches all contracts within the strike range and
        builds an OptionsChain object for analysis.

        Args:
            symbol: Underlying symbol
            snapshot_date: Date for the snapshot
            expiry: Option expiration date
            strike_range_pct: Filter to strikes within this % of spot
            underlying_price: Spot price (fetched if not provided)

        Returns:
            OptionsChain with all contracts
        """
        # Get strikes for this expiry
        all_strikes = self.list_strikes(symbol, expiry)

        if not all_strikes:
            logger.warning(f"No strikes found for {symbol} expiry {expiry}")
            return OptionsChain(
                symbol=symbol,
                date=snapshot_date,
                expiry=expiry,
                underlying_price=underlying_price or 0.0,
            )

        # Estimate underlying price from ATM strikes if not provided
        if underlying_price is None:
            # Use middle strike as estimate
            underlying_price = all_strikes[len(all_strikes) // 2]

        # Filter strikes to range
        min_strike = underlying_price * (1 - strike_range_pct)
        max_strike = underlying_price * (1 + strike_range_pct)
        strikes = [s for s in all_strikes if min_strike <= s <= max_strike]

        contracts = []

        # Fetch data for each strike
        for strike in strikes:
            for right in ["C", "P"]:
                try:
                    eod_data = self.get_eod_data(
                        symbol=symbol,
                        expiry=expiry,
                        strike=strike,
                        right=right,
                        start_date=snapshot_date,
                        end_date=snapshot_date,
                    )

                    if eod_data:
                        record = eod_data[0] if isinstance(eod_data, list) else eod_data
                        snapshot = self._parse_eod_record(
                            record, symbol, snapshot_date, expiry, strike, right
                        )
                        if snapshot:
                            contracts.append(snapshot)

                except ThetaDataError as e:
                    logger.debug(
                        f"No data for {symbol} {expiry} {strike} {right}: {e}"
                    )
                    continue

        return OptionsChain(
            symbol=symbol,
            date=snapshot_date,
            expiry=expiry,
            underlying_price=underlying_price,
            contracts=contracts,
        )

    def _parse_eod_record(
        self,
        record: Dict[str, Any],
        symbol: str,
        snapshot_date: date,
        expiry: date,
        strike: float,
        right: str,
    ) -> Optional[OptionSnapshot]:
        """
        Parse EOD record into OptionSnapshot.

        Args:
            record: Raw API record
            symbol: Underlying symbol
            snapshot_date: Data date
            expiry: Expiration date
            strike: Strike price
            right: 'C' or 'P'

        Returns:
            OptionSnapshot or None if parsing fails
        """
        try:
            # ThetaData EOD response fields (indices may vary)
            # Typical: [date, open, high, low, close, volume, oi, bid, ask, ...]

            # Handle different response formats
            if isinstance(record, dict):
                oi = record.get("open_interest", record.get("oi", 0))
                volume = record.get("volume", 0)
                bid = record.get("bid", 0.0)
                ask = record.get("ask", 0.0)
                underlying = record.get("underlying_price", 0.0)
            elif isinstance(record, (list, tuple)):
                # Array format - positions depend on endpoint
                # EOD: [ms_of_day, open, high, low, close, volume, count, bid, ask, underlying, oi]
                if len(record) >= 11:
                    volume = int(record[5]) if record[5] else 0
                    bid = float(record[7]) if record[7] else 0.0
                    ask = float(record[8]) if record[8] else 0.0
                    underlying = float(record[9]) if record[9] else 0.0
                    oi = int(record[10]) if record[10] else 0
                else:
                    return None
            else:
                return None

            return OptionSnapshot(
                symbol=symbol,
                date=snapshot_date,
                expiry=expiry,
                strike=strike,
                option_type=OptionRight(right),
                open_interest=oi,
                volume=volume,
                bid=bid,
                ask=ask,
                underlying_price=underlying,
            )

        except Exception as e:
            logger.debug(f"Failed to parse EOD record: {e}")
            return None

    def get_historical_chains(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        expiry_type: str = "monthly",
        strike_range_pct: float = 0.15,
    ) -> Dict[Tuple[date, date], OptionsChain]:
        """
        Fetch historical options chains for a date range.

        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            expiry_type: 'monthly', 'weekly', or 'all'
            strike_range_pct: Filter strikes within this % of spot

        Returns:
            Dict mapping (snapshot_date, expiry) to OptionsChain
        """
        chains = {}

        # Get all expirations in range
        expirations = self.list_expirations(symbol, start_date, end_date)

        # Filter to monthly expirations if requested
        if expiry_type == "monthly":
            expirations = self._filter_monthly_expirations(expirations)

        logger.info(
            f"Fetching chains for {symbol}: {len(expirations)} expirations, "
            f"{start_date} to {end_date}"
        )

        # For each expiration, get chains for relevant dates
        for expiry in expirations:
            # Get data for dates leading up to expiration
            # Typically interested in T-5 to T+2 around OpEx
            chain_start = max(start_date, expiry - timedelta(days=10))
            chain_end = min(end_date, expiry + timedelta(days=5))

            current = chain_start
            while current <= chain_end:
                try:
                    chain = self.get_options_chain(
                        symbol=symbol,
                        snapshot_date=current,
                        expiry=expiry,
                        strike_range_pct=strike_range_pct,
                    )

                    if chain.contracts:
                        chains[(current, expiry)] = chain
                        logger.debug(
                            f"Fetched {len(chain.contracts)} contracts for "
                            f"{symbol} {current} expiry {expiry}"
                        )

                except ThetaDataError as e:
                    logger.warning(f"Failed to get chain {symbol} {current} {expiry}: {e}")

                current += timedelta(days=1)

        logger.info(f"Fetched {len(chains)} chains for {symbol}")
        return chains

    def _filter_monthly_expirations(self, expirations: List[date]) -> List[date]:
        """
        Filter to monthly (third Friday) expirations only.

        Args:
            expirations: List of all expiration dates

        Returns:
            List of monthly expirations only
        """
        monthly = []
        for exp in expirations:
            # Check if this is a third Friday
            if exp.weekday() == 4:  # Friday
                # Find first day of month
                first_of_month = exp.replace(day=1)
                # Find first Friday
                days_to_friday = (4 - first_of_month.weekday()) % 7
                first_friday = first_of_month + timedelta(days=days_to_friday)
                # Third Friday
                third_friday = first_friday + timedelta(days=14)

                if exp == third_friday:
                    monthly.append(exp)

        return monthly

    def to_dataframe(self, chain: OptionsChain) -> pd.DataFrame:
        """
        Convert OptionsChain to pandas DataFrame.

        Args:
            chain: OptionsChain object

        Returns:
            DataFrame with one row per contract
        """
        if not chain.contracts:
            return pd.DataFrame()

        records = []
        for contract in chain.contracts:
            record = {
                "symbol": contract.symbol,
                "date": contract.date,
                "expiry": contract.expiry,
                "strike": contract.strike,
                "option_type": contract.option_type.value,
                "open_interest": contract.open_interest,
                "volume": contract.volume,
                "bid": contract.bid,
                "ask": contract.ask,
                "underlying_price": contract.underlying_price,
                "delta": contract.delta,
                "gamma": contract.gamma,
                "theta": contract.theta,
                "vega": contract.vega,
                "implied_vol": contract.implied_vol,
                "days_to_expiry": contract.days_to_expiry,
            }
            records.append(record)

        return pd.DataFrame(records)
