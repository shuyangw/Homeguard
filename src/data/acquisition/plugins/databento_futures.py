"""Databento GLBX.MDP3 futures plugin - downloads trade data and reconstructs OHLCV-1m."""

import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.data.acquisition.aggregators import trades_to_ohlcv_1m
from src.data.acquisition.base import BaseDownloader, DownloadResult
from src.data.acquisition.schemas import CANONICAL_OHLCV_SCHEMA, FUTURES_TRADES_SCHEMA
from src.utils.logger import get_logger

try:
    import databento as db
except ImportError:
    db = None  # type: ignore[assignment]

logger = get_logger(__name__)

DEFAULT_FUTURES_UNIVERSE = ["ES", "NQ", "CL", "GC", "ZN", "6E", "ZC", "YM", "RTY"]

DATASET = "GLBX.MDP3"
DEFAULT_START = "2020-01-01"


class DatabentoFuturesPlugin(BaseDownloader):
    """Downloads raw trade data from Databento and reconstructs OHLCV-1m bars.

    Two-stage storage:
    - futures_trades/symbol=ES/year=Y/month=M/data.parquet (raw trades)
    - futures_1min/symbol=ES/year=Y/month=M/data.parquet (reconstructed OHLCV)
    """

    def __init__(self, output_dir: Optional[Path] = None, **kwargs):
        self._api_key = os.getenv("DATABENTO_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DATABENTO_API_KEY not set in environment. "
                "Add it to your .env file."
            )
        if db is None:
            raise ImportError(
                "databento package not installed. Run: pip install databento"
            )
        super().__init__(output_dir=output_dir, **kwargs)

    def _create_client(self) -> Any:
        return db.Historical(self._api_key)

    def _fetch_symbol_data(
        self, client: Any, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        api_symbol = self._to_api_symbol(symbol)
        logger.info(f"Fetching trades for {api_symbol} from {start} to {end}")

        data = client.timeseries.get_range(
            dataset=DATASET,
            schema="trades",
            stype_in="continuous",
            symbols=[api_symbol],
            start=start,
            end=end,
        )
        df = data.to_df()

        if df.empty:
            return pd.DataFrame(columns=FUTURES_TRADES_SCHEMA)

        # Normalize Databento DataFrame to our trades schema
        # Databento sets ts_event as the index by default
        ts_series = (
            df.index
            if df.index.name == "ts_event"
            else df["ts_event"]
        )
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_series, utc=True),
                "price": df["price"].astype(float),
                "size": df["size"].astype(float),
            }
        )

        return result

    def _get_schema(self) -> list[str]:
        return FUTURES_TRADES_SCHEMA

    def _get_storage_subdir(self) -> str:
        return "futures_trades"

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol

    def _to_api_symbol(self, symbol: str) -> str:
        """Convert short symbol (ES) to Databento continuous format (ES.c.0)."""
        if ".c." not in symbol:
            return f"{symbol}.c.0"
        return symbol

    def download(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = DEFAULT_START,
        end_date: Optional[str] = None,
        skip_existing: bool = False,
    ) -> DownloadResult:
        """Download trades and reconstruct OHLCV-1m bars.

        Overrides base to add OHLCV reconstruction after raw trade storage.
        """
        symbols = symbols or DEFAULT_FUTURES_UNIVERSE

        # Step 1: Download raw trades using base infrastructure
        result = super().download(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            skip_existing=skip_existing,
        )

        # Step 2: Reconstruct OHLCV-1m from raw trades
        self._reconstruct_ohlcv(symbols)

        return result

    def _reconstruct_ohlcv(self, symbols: List[str]) -> None:
        """Read raw trades and write OHLCV-1m parquet files."""
        trades_dir = self.base_output_dir / "futures_trades"
        ohlcv_dir = self.base_output_dir / "futures_1min"

        for symbol in symbols:
            fs_symbol = self._normalize_symbol(symbol)
            symbol_trades_dir = trades_dir / f"symbol={fs_symbol}"
            if not symbol_trades_dir.exists():
                continue

            for parquet_file in symbol_trades_dir.glob("**/data.parquet"):
                # Extract year/month from path
                parts = parquet_file.parts
                year_part = [p for p in parts if p.startswith("year=")]
                month_part = [p for p in parts if p.startswith("month=")]
                if not year_part or not month_part:
                    continue

                year = year_part[0].replace("year=", "")
                month = month_part[0].replace("month=", "")

                # Read trades and reconstruct
                trades_df = pd.read_parquet(parquet_file)
                if trades_df.empty:
                    continue

                ohlcv_df = trades_to_ohlcv_1m(trades_df)
                if ohlcv_df.empty:
                    continue

                # Write to futures_1min with same partition structure
                out_dir = (
                    ohlcv_dir
                    / f"symbol={fs_symbol}"
                    / f"year={year}"
                    / f"month={month}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                ohlcv_df.to_parquet(out_dir / "data.parquet", index=False)

            logger.info(f"Reconstructed OHLCV-1m for {symbol}")
