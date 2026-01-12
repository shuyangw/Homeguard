"""
ThetaData Parquet Adapter.

Reads options data from ThetaData's partitioned parquet format
and converts to the OpEx strategy schema (OptionsChain, OptionSnapshot).

The ThetaData format stores 1-minute options data partitioned by:
  options_1min/root={symbol}/year={YYYY}/month={MM}/day_{DD}.parquet

This adapter:
1. Loads daily parquet files
2. Extracts EOD snapshots (last bar of each contract per day)
3. Converts to OptionsChain for GEX calculations
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

from src.data.options.options_schema import (
    OptionSnapshot,
    OptionsChain,
    OptionRight,
    DailyGEXSummary,
)
from src.settings import get_options_data_dir
from src.utils.logger import logger
from scipy.stats import norm
import math


class ThetaDataAdapter:
    """
    Adapter to read ThetaData parquet files and convert to OpEx schema.

    The adapter extracts EOD (end-of-day) snapshots from 1-minute data
    for use in daily GEX calculations.

    Note: ThetaData 1-min data may not include gamma and open_interest.
    When missing, gamma is estimated from Black-Scholes and OI is approximated
    from cumulative volume (rough proxy for testing purposes).
    """

    # Risk-free rate assumption for gamma estimation
    RISK_FREE_RATE = 0.05

    def __init__(self, data_dir: Optional[Path] = None, estimate_gamma: bool = True):
        """
        Initialize the adapter.

        Args:
            data_dir: Root options data directory. Defaults to settings.
            estimate_gamma: If True, estimate gamma when missing from data
        """
        self.data_dir = Path(data_dir) if data_dir else Path(get_options_data_dir())
        self.options_1min_dir = self.data_dir / "options_1min"
        self.estimate_gamma = estimate_gamma

        if not self.options_1min_dir.exists():
            logger.warning(f"Options data directory not found: {self.options_1min_dir}")

    def _estimate_gamma_bs(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        implied_vol: float,
        is_call: bool = True,
    ) -> float:
        """
        Estimate gamma using Black-Scholes formula.

        gamma = N'(d1) / (S * sigma * sqrt(T))

        Args:
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            implied_vol: Implied volatility (decimal)
            is_call: True for call, False for put (gamma is same for both)

        Returns:
            Estimated gamma
        """
        if time_to_expiry <= 0 or implied_vol <= 0 or spot <= 0 or strike <= 0:
            return 0.0

        try:
            sqrt_t = math.sqrt(time_to_expiry)
            d1 = (math.log(spot / strike) + (self.RISK_FREE_RATE + 0.5 * implied_vol**2) * time_to_expiry) / (implied_vol * sqrt_t)

            # Standard normal PDF at d1
            n_prime_d1 = norm.pdf(d1)

            # Gamma formula
            gamma = n_prime_d1 / (spot * implied_vol * sqrt_t)
            return gamma
        except Exception:
            return 0.0

    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols with available data.

        Returns:
            List of symbol strings (e.g., ['SPY', 'QQQ', 'IWM'])
        """
        if not self.options_1min_dir.exists():
            return []

        symbols = []
        for path in self.options_1min_dir.iterdir():
            if path.is_dir() and path.name.startswith("root="):
                symbol = path.name.replace("root=", "")
                symbols.append(symbol)

        return sorted(symbols)

    def get_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """
        Get available date range for a symbol.

        Args:
            symbol: The underlying symbol

        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data
        """
        symbol_dir = self.options_1min_dir / f"root={symbol}"
        if not symbol_dir.exists():
            return None, None

        dates = []
        for year_dir in symbol_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.replace("year=", ""))

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.replace("month=", ""))

                for file in month_dir.iterdir():
                    if file.suffix == ".parquet":
                        if file.name.startswith("day_"):
                            day = int(file.stem.replace("day_", ""))
                            dates.append(date(year, month, day))
                        elif file.name == "data.parquet":
                            # Monthly aggregate file - need to read to get dates
                            pass

        if not dates:
            return None, None

        return min(dates), max(dates)

    def _get_parquet_path(self, symbol: str, d: date) -> Optional[Path]:
        """Get path to parquet file for a specific date."""
        # Try daily file first
        daily_path = (
            self.options_1min_dir / f"root={symbol}" /
            f"year={d.year}" / f"month={d.month:02d}" / f"day_{d.day:02d}.parquet"
        )
        if daily_path.exists():
            return daily_path

        # Try monthly aggregate
        monthly_path = (
            self.options_1min_dir / f"root={symbol}" /
            f"year={d.year}" / f"month={d.month:02d}" / "data.parquet"
        )
        if monthly_path.exists():
            return monthly_path

        return None

    def _read_parquet_safe(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Read parquet file with dictionary column handling.

        ThetaData uses uint32 dictionary indices which pandas doesn't support,
        so we cast them to their value types.
        """
        try:
            table = pq.read_table(path)

            # Cast dictionary columns to value types
            new_cols = {}
            for field in table.schema:
                col = table.column(field.name)
                if pa.types.is_dictionary(field.type):
                    new_cols[field.name] = col.cast(field.type.value_type)
                else:
                    new_cols[field.name] = col

            new_table = pa.table(new_cols)
            return new_table.to_pandas()

        except Exception as e:
            logger.error(f"Error reading parquet {path}: {e}")
            return None

    def load_eod_chain(
        self,
        symbol: str,
        d: date,
        expiry_filter: Optional[date] = None,
    ) -> Optional[OptionsChain]:
        """
        Load EOD options chain for a symbol and date.

        Extracts the last available snapshot of each contract
        (approximating EOD data from 1-minute bars).

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            d: The date to load
            expiry_filter: If provided, only include this expiration

        Returns:
            OptionsChain object or None if no data
        """
        path = self._get_parquet_path(symbol, d)
        if path is None:
            return None

        df = self._read_parquet_safe(path)
        if df is None or df.empty:
            return None

        # Filter to the specific date if reading from monthly file
        if 'timestamp' in df.columns:
            df['ts_date'] = pd.to_datetime(df['timestamp']).dt.date
            df = df[df['ts_date'] == d]
            if df.empty:
                return None

        # Apply expiry filter
        if expiry_filter is not None:
            df = df[df['expiration'] == expiry_filter]
            if df.empty:
                return None

        # Get EOD snapshot: last bar of each contract
        # Group by expiry + strike + right, take last row
        eod_df = (
            df.sort_values('timestamp')
            .groupby(['expiration', 'strike', 'right'])
            .last()
            .reset_index()
        )

        if eod_df.empty:
            return None

        # Get underlying price (should be consistent across contracts)
        underlying_price = eod_df['underlying_px'].dropna().iloc[-1] if 'underlying_px' in eod_df.columns else 0.0

        # Determine expiry for the chain (use nearest monthly if no filter)
        if expiry_filter:
            chain_expiry = expiry_filter
        else:
            # Find the nearest monthly expiry (3rd Friday)
            chain_expiry = self._find_nearest_monthly_expiry(d, eod_df['expiration'].unique())

        # Build OptionSnapshot objects
        contracts = []
        for _, row in eod_df.iterrows():
            try:
                # Get values, handling potential NaN/missing
                strike = float(row['strike'])
                expiry_date = row['expiration']
                # ThetaData uses 'CALL'/'PUT', convert to 'C'/'P'
                raw_right = row['right']
                if raw_right in ('CALL', 'C'):
                    option_right = 'C'
                elif raw_right in ('PUT', 'P'):
                    option_right = 'P'
                else:
                    continue  # Unknown option type
                spot = float(row.get('underlying_px', underlying_price) or underlying_price)
                implied_vol = float(row.get('implied_vol', 0) or 0)
                volume = int(row.get('volume', 0) or 0)

                # Handle gamma - estimate if missing
                gamma_val = row.get('gamma')
                if pd.isna(gamma_val) and self.estimate_gamma:
                    # Estimate gamma from Black-Scholes
                    days_to_exp = (expiry_date - d).days if hasattr(expiry_date, 'days') else (pd.Timestamp(expiry_date).date() - d).days
                    time_to_expiry = max(days_to_exp / 365.0, 0.001)  # Avoid zero
                    gamma_val = self._estimate_gamma_bs(spot, strike, time_to_expiry, implied_vol)
                elif pd.isna(gamma_val):
                    gamma_val = 0.0
                else:
                    gamma_val = float(gamma_val)

                # Handle open interest - use volume as proxy if missing
                oi_val = row.get('open_interest')
                if pd.isna(oi_val) or oi_val <= 0:
                    # Use daily volume as rough OI proxy (scaled up)
                    # Typical OI is 5-10x daily volume for liquid options
                    oi_val = max(volume * 5, 100)  # Minimum 100 contracts
                else:
                    oi_val = int(oi_val)

                # Skip if still no meaningful data
                if gamma_val == 0 and oi_val <= 0:
                    continue

                snapshot = OptionSnapshot(
                    symbol=symbol,
                    date=d,
                    expiry=expiry_date if isinstance(expiry_date, date) else pd.Timestamp(expiry_date).date(),
                    strike=strike,
                    option_type=OptionRight(option_right),
                    open_interest=oi_val,
                    volume=volume,
                    bid=float(row.get('bid_close', 0) or 0),
                    ask=float(row.get('ask_close', 0) or 0),
                    underlying_price=spot,
                    delta=float(row.get('delta', 0) or 0),
                    gamma=gamma_val,
                    theta=float(row.get('theta', 0) or 0),
                    vega=float(row.get('vega', 0) or 0),
                    implied_vol=implied_vol,
                )
                contracts.append(snapshot)
            except Exception as e:
                logger.debug(f"Skipping contract row: {e}")
                continue

        if not contracts:
            return None

        return OptionsChain(
            symbol=symbol,
            date=d,
            expiry=chain_expiry,
            underlying_price=underlying_price,
            contracts=contracts,
        )

    def _find_nearest_monthly_expiry(self, d: date, available_expiries) -> date:
        """Find nearest monthly expiry (3rd Friday) from available expirations."""
        from src.strategies.opex.calendar import OpExCalendar

        calendar = OpExCalendar()

        # Get monthly expirations for this year and next
        monthly_exps = []
        for year in [d.year, d.year + 1]:
            monthly_exps.extend(calendar.get_monthly_expirations(year))

        # Find the one that's in our available expiries and nearest to date
        available_set = set()
        for exp in available_expiries:
            if hasattr(exp, 'date'):
                available_set.add(exp)
            elif isinstance(exp, date):
                available_set.add(exp)
            else:
                # Try converting via pandas
                try:
                    available_set.add(pd.Timestamp(exp).date())
                except Exception:
                    pass

        for exp in sorted(monthly_exps):
            if exp in available_set and exp >= d:
                return exp

        # Fallback: return closest available expiry
        available_list = sorted(available_set)
        if available_list:
            return min(available_list, key=lambda x: abs((x - d).days))

        return d  # Last resort

    def load_chains_for_period(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        monthly_only: bool = True,
    ) -> Dict[date, OptionsChain]:
        """
        Load options chains for a date range.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period
            monthly_only: If True, filter to monthly expirations only

        Returns:
            Dict mapping dates to OptionsChain objects
        """
        chains = {}
        current = start_date

        while current <= end_date:
            chain = self.load_eod_chain(symbol, current)
            if chain is not None:
                chains[current] = chain
            current += timedelta(days=1)

        logger.info(f"Loaded {len(chains)} chains for {symbol} ({start_date} to {end_date})")
        return chains

    def get_trading_dates(self, symbol: str, start_date: date, end_date: date) -> List[date]:
        """
        Get list of dates with available data.

        Args:
            symbol: Underlying symbol
            start_date: Start of period
            end_date: End of period

        Returns:
            Sorted list of dates with data
        """
        symbol_dir = self.options_1min_dir / f"root={symbol}"
        if not symbol_dir.exists():
            return []

        dates = []
        for year_dir in symbol_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            year = int(year_dir.name.replace("year=", ""))

            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                    continue
                month = int(month_dir.name.replace("month=", ""))

                for file in month_dir.iterdir():
                    if file.suffix == ".parquet" and file.name.startswith("day_"):
                        day = int(file.stem.replace("day_", ""))
                        d = date(year, month, day)
                        if start_date <= d <= end_date:
                            dates.append(d)

        return sorted(dates)
