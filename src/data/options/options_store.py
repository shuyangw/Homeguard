"""
Options data storage using Parquet files.

Stores historical options chain data in a Hive-partitioned format
for efficient querying by symbol, date, and expiry.

Directory structure:
    {storage_dir}/options/
        symbol={SPY}/
            year={2024}/
                month={01}/
                    options_chains.parquet
        gex_daily/
            symbol={SPY}/
                gex_summary.parquet
"""

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.settings import get_local_storage_dir
from src.utils.logger import logger
from src.data.options.options_schema import (
    OptionSnapshot,
    OptionsChain,
    StrikeGEX,
    DailyGEXSummary,
    OptionRight,
)


class OptionsDataStore:
    """
    Parquet-based storage for historical options data.

    Stores options chains partitioned by symbol/year/month for
    efficient access patterns needed by OpEx strategy.

    Usage:
        store = OptionsDataStore()

        # Save chain
        store.save_chain(chain)

        # Load chain
        chain = store.get_chain("SPY", date(2024, 1, 15), date(2024, 1, 19))

        # Get chains for date range
        chains = store.get_chains_for_range("SPY", start, end, monthly_only=True)
    """

    # Parquet schema for options data
    OPTIONS_SCHEMA = pa.schema([
        ("symbol", pa.string()),
        ("date", pa.date32()),
        ("expiry", pa.date32()),
        ("strike", pa.float64()),
        ("option_type", pa.string()),
        ("open_interest", pa.int64()),
        ("volume", pa.int64()),
        ("bid", pa.float64()),
        ("ask", pa.float64()),
        ("underlying_price", pa.float64()),
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("theta", pa.float64()),
        ("vega", pa.float64()),
        ("implied_vol", pa.float64()),
        ("days_to_expiry", pa.int32()),
    ])

    GEX_SCHEMA = pa.schema([
        ("symbol", pa.string()),
        ("date", pa.date32()),
        ("expiry", pa.date32()),
        ("underlying_price", pa.float64()),
        ("total_gex", pa.float64()),
        ("call_gex", pa.float64()),
        ("put_gex", pa.float64()),
        ("pin_strike", pa.float64()),
        ("pin_strike_gex", pa.float64()),
        ("distance_to_pin", pa.float64()),
        ("total_oi", pa.int64()),
        ("put_call_ratio", pa.float64()),
        ("gex_percentile", pa.float64()),
    ])

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize options data store.

        Args:
            base_dir: Base directory for storage (default: {storage_dir}/options)
        """
        if base_dir is None:
            storage_dir = get_local_storage_dir()
            base_dir = os.path.join(storage_dir, "options")

        self.base_dir = Path(base_dir)
        self.chains_dir = self.base_dir / "chains"
        self.gex_dir = self.base_dir / "gex_daily"

        # Create directories
        self.chains_dir.mkdir(parents=True, exist_ok=True)
        self.gex_dir.mkdir(parents=True, exist_ok=True)

    def _get_chain_path(self, symbol: str, snapshot_date: date) -> Path:
        """
        Get path for options chain file.

        Partitioned by symbol/year/month for efficient queries.
        """
        return (
            self.chains_dir
            / f"symbol={symbol.upper()}"
            / f"year={snapshot_date.year}"
            / f"month={snapshot_date.month:02d}"
            / "chains.parquet"
        )

    def _get_gex_path(self, symbol: str) -> Path:
        """Get path for GEX summary file."""
        return self.gex_dir / f"symbol={symbol.upper()}" / "gex_summary.parquet"

    def save_chain(self, chain: OptionsChain) -> bool:
        """
        Save options chain to storage.

        Appends to existing file or creates new one.

        Args:
            chain: OptionsChain to save

        Returns:
            True if saved successfully
        """
        if not chain.contracts:
            logger.debug(f"Empty chain, not saving: {chain.symbol} {chain.date}")
            return False

        path = self._get_chain_path(chain.symbol, chain.date)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        records = []
        for contract in chain.contracts:
            records.append({
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
            })

        new_df = pd.DataFrame(records)

        # Load existing data and merge
        if path.exists():
            existing_df = pd.read_parquet(path)

            # Remove duplicates (same date/expiry/strike/type)
            key_cols = ["date", "expiry", "strike", "option_type"]
            existing_df = existing_df[
                ~existing_df.set_index(key_cols).index.isin(
                    new_df.set_index(key_cols).index
                )
            ]

            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df

        # Sort for better compression
        df = df.sort_values(["date", "expiry", "strike", "option_type"])

        # Save with compression
        df.to_parquet(path, index=False, compression="snappy")

        logger.debug(
            f"Saved {len(new_df)} contracts for {chain.symbol} "
            f"{chain.date} expiry {chain.expiry}"
        )
        return True

    def save_chains(self, chains: List[OptionsChain]) -> int:
        """
        Save multiple chains.

        Args:
            chains: List of OptionsChain objects

        Returns:
            Number of chains saved
        """
        count = 0
        for chain in chains:
            if self.save_chain(chain):
                count += 1
        return count

    def get_chain(
        self,
        symbol: str,
        snapshot_date: date,
        expiry: date,
    ) -> Optional[OptionsChain]:
        """
        Load options chain for specific symbol/date/expiry.

        Args:
            symbol: Underlying symbol
            snapshot_date: Date of snapshot
            expiry: Expiration date

        Returns:
            OptionsChain or None if not found
        """
        path = self._get_chain_path(symbol, snapshot_date)

        if not path.exists():
            return None

        df = pd.read_parquet(path)

        # Convert date columns to Python date objects for comparison
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date

        # Filter to specific date and expiry
        mask = (df["date"] == snapshot_date) & (df["expiry"] == expiry)
        filtered = df[mask]

        if filtered.empty:
            return None

        return self._dataframe_to_chain(filtered, symbol, snapshot_date, expiry)

    def get_chains_for_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        expiry: Optional[date] = None,
        monthly_only: bool = True,
    ) -> Dict[Tuple[date, date], OptionsChain]:
        """
        Load chains for a date range.

        Args:
            symbol: Underlying symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            expiry: Filter to specific expiry (optional)
            monthly_only: Only return monthly expirations

        Returns:
            Dict mapping (snapshot_date, expiry) to OptionsChain
        """
        chains = {}

        # Find all relevant parquet files
        pattern = f"symbol={symbol.upper()}/year=*/month=*/chains.parquet"
        files = list(self.chains_dir.glob(pattern))

        if not files:
            logger.debug(f"No data files found for {symbol}")
            return chains

        # Load and filter data
        all_data = []
        for file_path in files:
            df = pd.read_parquet(file_path)

            # Filter by date range
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date

            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            if expiry is not None:
                mask &= (df["expiry"] == expiry)

            filtered = df[mask]
            if not filtered.empty:
                all_data.append(filtered)

        if not all_data:
            return chains

        combined = pd.concat(all_data, ignore_index=True)

        # Filter to monthly expirations if requested
        if monthly_only:
            combined = combined[
                combined["expiry"].apply(self._is_monthly_expiration)
            ]

        # Group by (date, expiry) and create chains
        for (snap_date, exp), group in combined.groupby(["date", "expiry"]):
            chain = self._dataframe_to_chain(group, symbol, snap_date, exp)
            if chain and chain.contracts:
                chains[(snap_date, exp)] = chain

        return chains

    def _is_monthly_expiration(self, exp_date: date) -> bool:
        """Check if date is a monthly expiration (third Friday)."""
        if exp_date.weekday() != 4:  # Not Friday
            return False

        first_of_month = exp_date.replace(day=1)
        days_to_friday = (4 - first_of_month.weekday()) % 7
        first_friday = first_of_month + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(days=14)

        return exp_date == third_friday

    def _dataframe_to_chain(
        self,
        df: pd.DataFrame,
        symbol: str,
        snapshot_date: date,
        expiry: date,
    ) -> Optional[OptionsChain]:
        """Convert DataFrame rows to OptionsChain."""
        if df.empty:
            return None

        contracts = []
        underlying_price = 0.0

        for _, row in df.iterrows():
            contract = OptionSnapshot(
                symbol=row["symbol"],
                date=row["date"] if isinstance(row["date"], date) else row["date"].date(),
                expiry=row["expiry"] if isinstance(row["expiry"], date) else row["expiry"].date(),
                strike=row["strike"],
                option_type=OptionRight(row["option_type"]),
                open_interest=int(row["open_interest"]),
                volume=int(row.get("volume", 0)),
                bid=float(row.get("bid", 0)),
                ask=float(row.get("ask", 0)),
                underlying_price=float(row.get("underlying_price", 0)),
                delta=float(row.get("delta", 0)),
                gamma=float(row.get("gamma", 0)),
                theta=float(row.get("theta", 0)),
                vega=float(row.get("vega", 0)),
                implied_vol=float(row.get("implied_vol", 0)),
            )
            contracts.append(contract)

            if contract.underlying_price > 0:
                underlying_price = contract.underlying_price

        return OptionsChain(
            symbol=symbol,
            date=snapshot_date,
            expiry=expiry,
            underlying_price=underlying_price,
            contracts=contracts,
        )

    def save_gex_summary(self, summary: DailyGEXSummary) -> bool:
        """
        Save daily GEX summary.

        Args:
            summary: DailyGEXSummary to save

        Returns:
            True if saved successfully
        """
        path = self._get_gex_path(summary.symbol)
        path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "symbol": summary.symbol,
            "date": summary.date,
            "expiry": summary.expiry,
            "underlying_price": summary.underlying_price,
            "total_gex": summary.total_gex,
            "call_gex": summary.call_gex,
            "put_gex": summary.put_gex,
            "pin_strike": summary.pin_strike,
            "pin_strike_gex": summary.pin_strike_gex,
            "distance_to_pin": summary.distance_to_pin,
            "total_oi": summary.total_oi,
            "put_call_ratio": summary.put_call_ratio,
            "gex_percentile": summary.gex_percentile,
        }

        new_df = pd.DataFrame([record])

        # Merge with existing
        if path.exists():
            existing_df = pd.read_parquet(path)
            key_cols = ["date", "expiry"]
            existing_df = existing_df[
                ~existing_df.set_index(key_cols).index.isin(
                    new_df.set_index(key_cols).index
                )
            ]
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df

        df = df.sort_values(["date", "expiry"])
        df.to_parquet(path, index=False, compression="snappy")

        return True

    def get_gex_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get GEX history for a symbol.

        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily GEX summaries
        """
        path = self._get_gex_path(symbol)

        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        return df[mask].sort_values("date")

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with stored data."""
        symbols = set()

        # Check chains directory
        for path in self.chains_dir.glob("symbol=*"):
            symbol = path.name.replace("symbol=", "")
            symbols.add(symbol)

        return sorted(symbols)

    def get_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """
        Get date range available for a symbol.

        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data
        """
        pattern = f"symbol={symbol.upper()}/year=*/month=*/chains.parquet"
        files = list(self.chains_dir.glob(pattern))

        if not files:
            return None, None

        min_date = None
        max_date = None

        for file_path in files:
            df = pd.read_parquet(file_path, columns=["date"])
            df["date"] = pd.to_datetime(df["date"]).dt.date

            file_min = df["date"].min()
            file_max = df["date"].max()

            if min_date is None or file_min < min_date:
                min_date = file_min
            if max_date is None or file_max > max_date:
                max_date = file_max

        return min_date, max_date

    def get_stats(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            Dict with storage stats
        """
        symbols = self.get_available_symbols()

        total_files = 0
        total_size_mb = 0.0

        for file_path in self.chains_dir.rglob("*.parquet"):
            total_files += 1
            total_size_mb += file_path.stat().st_size / (1024 * 1024)

        return {
            "symbols": len(symbols),
            "files": total_files,
            "size_mb": round(total_size_mb, 2),
            "chains_dir": str(self.chains_dir),
        }
